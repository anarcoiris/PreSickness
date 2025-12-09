---
description: Investigar avances en medicina predictiva y ML para diagnóstico
---

# Workflow: Investigación en Medicina Predictiva

Guía para investigar el estado del arte en predicción de enfermedades, especialmente
enfermedades autoinmunes como Esclerosis Múltiple.

## Fuentes Primarias

### Bases de Datos Científicas
1. **PubMed/MEDLINE**: https://pubmed.ncbi.nlm.nih.gov/
   - Queries sugeridas:
     - `"multiple sclerosis" AND "machine learning" AND "prediction"`
     - `"relapse prediction" AND "autoimmune"`
     - `"digital biomarkers" AND "neurological diseases"`

2. **Google Scholar**: https://scholar.google.com/
   - Filtrar por últimos 3 años
   - Buscar surveys y meta-análisis primero

3. **arXiv (cs.LG + stat.ML)**: https://arxiv.org/list/cs.LG/recent
   - Modelos temporales para salud
   - Time-series forecasting médico

4. **Nature Digital Medicine**: https://www.nature.com/npjdigitalmed/
   - Aplicaciones clínicas de ML

### Datasets Públicos
- **MIMIC-III/IV**: Datos de UCI (requiere credenciales)
- **UK Biobank**: Datos poblacionales
- **PhysioNet**: Señales fisiológicas
- **MS Base Registry**: Específico para EM (solicitar acceso)

## Áreas de Investigación

### 1. Biomarcadores Digitales
```
Keywords: digital phenotyping, passive sensing, smartphone biomarkers
Aplicación: Patrones de comunicación como proxy de estado cognitivo
```

### 2. Modelos Temporales para Salud
```
Keywords: temporal fusion transformer, health forecasting, disease progression
Modelos: TFT, N-BEATS, DeepAR, Informer
```

### 3. Análisis de Lenguaje Clínico
```
Keywords: clinical NLP, linguistic markers, depression detection
Métricas: perplexity, sentiment shift, lexical diversity
```

### 4. Predicción de Brotes en EM
```
Específico: MRI prediction, relapse risk, neurofilament light chain
Estado del arte: AUROC 0.65-0.75 en estudios publicados
```

## Papers Clave para Revisar

### Temporales y Forecasting
- [ ] "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (Lim et al., 2021)
- [ ] "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (Zhou et al., 2021)
- [ ] "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting" (Oreshkin et al., 2020)

### Predicción en Esclerosis Múltiple
- [ ] "Machine learning approaches for predicting relapse in multiple sclerosis" (buscar en PubMed)
- [ ] "Digital biomarkers in multiple sclerosis" (revisiones recientes)
- [ ] "Smartphone-based assessment in multiple sclerosis" (estudios de movilidad)

### NLP y Salud Mental
- [ ] "Language as a biomarker for psychosis" (Corcoran et al.)
- [ ] "Detecting depression from text" (meta-análisis)
- [ ] "Linguistic features predicting cognitive decline"

## Checklist de Investigación

### Fase 1: Scoping (1-2 horas)
- [ ] Buscar 3-5 surveys recientes en el área
- [ ] Identificar grupos de investigación líderes
- [ ] Listar datasets disponibles y sus requisitos

### Fase 2: Deep Dive (4-8 horas)
- [ ] Leer papers clave completos
- [ ] Extraer arquitecturas y métricas
- [ ] Documentar gaps en la literatura

### Fase 3: Síntesis
- [ ] Crear tabla comparativa de enfoques
- [ ] Identificar oportunidades de mejora
- [ ] Proponer experimentos basados en hallazgos

## Output Esperado

Crear documento en `docs/research/` con:
```markdown
# [Tema de Investigación]

## Resumen Ejecutivo
[2-3 párrafos]

## Estado del Arte
[Tabla comparativa]

## Gaps Identificados
[Lista]

## Propuesta
[Cómo aplicar a EM-Predictor]

## Referencias
[BibTeX o links]
```

## Herramientas Útiles

- **Semantic Scholar**: https://www.semanticscholar.org/ (API gratuita)
- **Connected Papers**: https://www.connectedpapers.com/ (grafo de citas)
- **Elicit**: https://elicit.org/ (AI para research)
- **Consensus**: https://consensus.app/ (búsqueda con LLM)
