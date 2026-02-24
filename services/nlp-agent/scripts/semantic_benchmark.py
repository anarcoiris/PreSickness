import asyncio
import httpx
import json
import os
from typing import List, Dict
import pandas as pd

# OpenRouter / OpenAI configuration
API_KEY = os.getenv("OPENAI_API_KEY")
NLP_AGENT_URL = "http://localhost:8000/v1/process"

SYMPTOMS = ["pain", "fatigue", "anxiety", "mood", "sleep"]

PROMPT_TEMPLATE = """
Genera 10 frases cortas en español (tipo WhatsApp o mensaje de diario) que expresen {symptom} de forma {intensity}. 
Las frases deben ser variadas, informales y realistas.
Devuelve SOLO una lista JSON de strings.
Ejemplo para dolor fuerte: ["No aguanto la espalda", "Me punza mucho la pierna"]
"""

async def generate_variants(symptom: str, intensity: str) -> List[str]:
    if not API_KEY:
        # Heuristic fallback if no API key
        placeholders = {
            "pain": ["Me duele {part}", "Tengo un pinchazo en {part}", "No soporto el dolor"],
            "fatigue": ["Estoy agotado", "No tengo fuerzas", "Siento mucha pesadez"],
            "anxiety": ["Estoy muy nervioso", "Siento angustia", "No puedo dejar de pensar"],
            "sleep": ["No he dormido nada", "Me desperté mil veces", "Pesadillas otra vez"],
            "mood": ["Estoy muy triste", "No tengo ganas de nada", "Qué día más gris"]
        }
        return [p.format(part="la cabeza") for p in placeholders.get(symptom, ["Test"])]

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(symptom=symptom, intensity=intensity)}],
                    "temperature": 0.7
                },
                timeout=30.0
            )
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            return json.loads(content)
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return []

async def test_phrase(client, phrase: str, expected_symptom: str) -> Dict:
    payload = {
        "message_id": "bench",
        "user_id": "eval_user",
        "text": phrase
    }
    try:
        resp = await client.post(NLP_AGENT_URL, json=payload)
        if resp.status_code == 200:
            data = resp.json()["symptom_scores"]
            return {
                "phrase": phrase,
                "expected": expected_symptom,
                **{s: data[s]["prob"] for s in SYMPTOMS}
            }
    except Exception as e:
        print(f"Error testing phrase '{phrase}': {e}")
    return None

async def main():
    print("--- Inciando Benchmark Semántico Avanzado ---")
    all_tests = []
    
    # 1. Generar datos
    print("Generando datos de prueba con GPT...")
    tasks = []
    for sym in SYMPTOMS:
        tasks.append(generate_variants(sym, "notable/fuerte"))
    
    results = await asyncio.gather(*tasks)
    
    test_cases = []
    for i, sym in enumerate(SYMPTOMS):
        for phrase in results[i]:
            test_cases.append((phrase, sym))
    
    print(f"Total frases generadas: {len(test_cases)}")
    
    # 2. Ejecutar Benchmark
    async with httpx.AsyncClient() as client:
        results = []
        for phrase, sym in test_cases:
            res = await test_phrase(client, phrase, sym)
            if res:
                results.append(res)
    
    # 3. Analizar resultados
    df = pd.DataFrame(results)
    print("\n--- Resultados Promedio por Categoría Esperada ---")
    summary = df.groupby("expected")[SYMPTOMS].mean()
    print(summary)
    
    # Verificar diagonal (sensibilidad)
    print("\n--- Diagnóstico de Sensibilidad ---")
    for sym in SYMPTOMS:
        avg_score = summary.loc[sym, sym]
        print(f"Sintoma: {sym:8} | Score Promedio: {avg_score:.4f} " + ("✅" if avg_score > 0.4 else "⚠️ Baja sensibilidad"))

    # Exportar para revisión
    df.to_csv("nlp_semantic_benchmark.csv", index=False)
    print(f"\nBenchmark guardado en nlp_semantic_benchmark.csv")

if __name__ == "__main__":
    asyncio.run(main())
