# Documentación Legal y Compliance

## Resumen

EM-Predictor procesa **datos de salud sensibles** y debe cumplir con:
- GDPR (Europa)
- HIPAA (EE.UU., si aplica)
- Legislación local de protección de datos

---

## Evaluación de Impacto (DPIA)

### Naturaleza del Procesamiento
- **Tipo**: Datos de categoría especial (salud)
- **Volumen**: Comunicaciones diarias durante meses/años
- **Fuente**: Exports de WhatsApp/Telegram del paciente

### Riesgos Identificados

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Acceso no autorizado | Media | Alto | Encriptación, autenticación |
| Re-identificación | Baja | Alto | Pseudonimización, no almacenar nombres |
| Uso secundario | Baja | Alto | Políticas de retención, consentimiento |
| Fuga de datos | Baja | Crítico | Modelos locales, sin APIs externas |

### Medidas Técnicas

1. **Encriptación en reposo**: AES-256 para todos los archivos de pacientes
2. **Encriptación en tránsito**: TLS 1.3 obligatorio
3. **Pseudonimización**: IDs hash en lugar de nombres reales
4. **Acceso mínimo**: Solo personal autorizado con logs de acceso
5. **Modelos locales**: Procesamiento sin enviar datos a terceros

---

## Consentimiento Informado

### Requisitos
El paciente debe consentir explícitamente:
- [ ] Procesamiento de sus comunicaciones para predicción
- [ ] Almacenamiento de datos durante el estudio
- [ ] Uso de resultados para investigación (anonimizados)

### Template de Consentimiento

```
CONSENTIMIENTO INFORMADO - ESTUDIO EM-PREDICTOR

Yo, ________________, acepto participar voluntariamente en el estudio 
EM-Predictor para la predicción de brotes de Esclerosis Múltiple.

ENTIENDO QUE:
1. Mis comunicaciones de WhatsApp serán analizadas por un sistema de IA
2. No se almacenarán nombres ni información identificable
3. Puedo retirarme en cualquier momento sin consecuencias
4. Los datos serán procesados de forma segura y local

AUTORIZO:
□ El análisis de mis mensajes para detectar patrones lingüísticos
□ El almacenamiento seguro de datos durante el estudio
□ El uso de resultados anonimizados para publicaciones científicas

Firma: ________________  Fecha: ________________
```

---

## Derechos del Interesado

### Derecho de Acceso
El paciente puede solicitar:
- Copia de todos sus datos almacenados
- Explicación del procesamiento realizado

### Derecho de Rectificación
El paciente puede corregir:
- Datos incorrectos en su perfil
- Eventos clínicos mal etiquetados

### Derecho al Olvido
El paciente puede solicitar:
- Eliminación completa de sus datos
- Confirmación de destrucción

**Implementación**:
```bash
python scripts/admin/delete_patient.py --patient-id <ID> --confirm
```

### Derecho a la Portabilidad
El paciente puede exportar sus datos en formato estándar (JSON/CSV).

---

## Retención de Datos

| Tipo de Dato | Período | Justificación |
|--------------|---------|---------------|
| Mensajes crudos | 0 días | No se almacenan después del procesamiento |
| Features procesados | 2 años | Duración del estudio |
| Modelos entrenados | 5 años | Validación científica |
| Logs de acceso | 1 año | Auditoría |

---

## Registro de Actividades

El sistema mantiene logs de:
- Accesos a datos de pacientes (quién, cuándo, qué)
- Predicciones generadas
- Modificaciones de datos

Logs almacenados sin PII y retenidos 1 año.

---

## Contacto DPO

Para ejercer derechos o consultas:
- Email: dpo@em-predictor.org
- Responsable: [Nombre del DPO]
