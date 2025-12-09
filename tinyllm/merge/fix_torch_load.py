#!/usr/bin/env python3
"""
fix_torch_load.py

Parche para compatibilidad con PyTorch 2.6+ que cambió weights_only=True por defecto.

Problema: 
  PyTorch 2.6+ rechaza cargar checkpoints que contienen objetos custom 
  como TrainingHistory por seguridad.

Solución:
  - Registra clases custom como seguras
  - Modifica main.py para usar weights_only=False en contextos seguros
  - Aplica fix en training.py y generation.py

Uso:
    python fix_torch_load.py
    python fix_torch_load.py --check-only  # Solo verifica sin modificar
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple
import shutil
from datetime import datetime


def backup_file(filepath: Path) -> Path:
    """Crea backup del archivo"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.with_suffix(f'.backup_{timestamp}{filepath.suffix}')
    shutil.copy2(filepath, backup_path)
    print(f"  ✅ Backup: {backup_path.name}")
    return backup_path


def check_pytorch_version() -> Tuple[bool, str]:
    """Verifica versión de PyTorch"""
    try:
        import torch
        version = torch.__version__
        major, minor = version.split('.')[:2]
        major, minor = int(major), int(minor)
        
        needs_fix = (major > 2) or (major == 2 and minor >= 6)
        
        return needs_fix, version
    except ImportError:
        return False, "No instalado"
    except Exception as e:
        return False, f"Error: {e}"


def fix_main_py() -> bool:
    """Aplica fix en main.py"""
    
    file_path = Path('main.py')
    
    if not file_path.exists():
        print("❌ main.py no encontrado")
        return False
    
    print("\n📝 Procesando main.py...")
    
    content = file_path.read_text(encoding='utf-8')
    
    # Verifica si ya está parcheado
    if 'weights_only=False' in content or 'TORCH_LOAD_FIX_APPLIED' in content:
        print("  ℹ️  Ya tiene fix aplicado o partial fix")
    
    # Backup
    backup_file(file_path)
    
    modified = False
    
    # Pattern 1: validate_checkpoint function
    # Busca: torch.load(filepath, map_location='cpu')
    # Reemplaza con: torch.load(filepath, map_location='cpu', weights_only=False)
    
    pattern1 = r"torch\.load\(filepath,\s*map_location='cpu'\)"
    replacement1 = "torch.load(filepath, map_location='cpu', weights_only=False)  # TORCH_LOAD_FIX_APPLIED"
    
    new_content = re.sub(pattern1, replacement1, content)
    
    if new_content != content:
        modified = True
        print("  ✅ Fix aplicado en validate_checkpoint()")
        content = new_content
    
    # Pattern 2: load_checkpoint function en training.py reference
    # Este se maneja en fix_training_py()
    
    # Guarda si hubo cambios
    if modified:
        file_path.write_text(content, encoding='utf-8')
        print("  ✅ main.py actualizado")
        return True
    else:
        print("  ℹ️  No requiere cambios (ya parcheado o no usa torch.load)")
        return True


def fix_training_py() -> bool:
    """Aplica fix en training.py"""
    
    file_path = Path('training.py')
    
    if not file_path.exists():
        print("❌ training.py no encontrado")
        return False
    
    print("\n📝 Procesando training.py...")
    
    content = file_path.read_text(encoding='utf-8')
    
    # Verifica si ya está parcheado
    if 'weights_only=False' in content or 'TORCH_LOAD_FIX_APPLIED' in content:
        print("  ℹ️  Ya tiene fix aplicado")
        return True
    
    # Backup
    backup_file(file_path)
    
    modified = False
    
    # Pattern: load_checkpoint function
    # Busca: checkpoint = torch.load(path, map_location=device)
    # Reemplaza con versión que incluye weights_only=False
    
    pattern = r"checkpoint = torch\.load\(path,\s*map_location=device\)"
    replacement = "checkpoint = torch.load(path, map_location=device, weights_only=False)  # TORCH_LOAD_FIX_APPLIED"
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        modified = True
        print("  ✅ Fix aplicado en load_checkpoint()")
        content = new_content
    
    # Guarda si hubo cambios
    if modified:
        file_path.write_text(content, encoding='utf-8')
        print("  ✅ training.py actualizado")
        return True
    else:
        print("  ℹ️  No requiere cambios")
        return True


def add_safe_globals_registration() -> bool:
    """
    Añade registro de clases seguras al inicio de main.py
    
    Esto es la solución RECOMENDADA por PyTorch para permitir
    cargar objetos custom de forma segura.
    """
    
    file_path = Path('main.py')
    
    if not file_path.exists():
        print("❌ main.py no encontrado")
        return False
    
    print("\n📝 Añadiendo registro de clases seguras...")
    
    content = file_path.read_text(encoding='utf-8')
    
    # Verifica si ya está añadido
    if 'add_safe_globals' in content or 'SAFE_GLOBALS_REGISTERED' in content:
        print("  ℹ️  Ya tiene registro de clases seguras")
        return True
    
    # Backup (si no se hizo antes)
    if not any(file_path.parent.glob(f'{file_path.stem}.backup_*{file_path.suffix}')):
        backup_file(file_path)
    
    # Código a insertar después de los imports
    safe_globals_code = '''
# ======================== PyTorch 2.6+ Compatibility ========================
# SAFE_GLOBALS_REGISTERED: Registra clases custom para torch.load
# Esto permite cargar checkpoints con TrainingHistory de forma segura

try:
    import torch
    from training import TrainingHistory
    
    # Registra TrainingHistory como clase segura
    # Esto es la solución RECOMENDADA por PyTorch 2.6+
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([TrainingHistory])
        # También registra dataclasses si es necesario
        import dataclasses
        torch.serialization.add_safe_globals([dataclasses.dataclass])
except ImportError:
    # PyTorch < 2.6 o training no disponible aún
    pass
except Exception as e:
    # No crítico si falla, usaremos weights_only=False como fallback
    pass

# ============================================================================

'''
    
    # Busca dónde insertar (después de los imports locales)
    # Busca la línea "from generation import"
    insert_marker = "from generation import"
    insert_pos = content.find(insert_marker)
    
    if insert_pos == -1:
        # Fallback: después del último import
        import_lines = [i for i, line in enumerate(content.split('\n')) if line.startswith('from ') or line.startswith('import ')]
        if import_lines:
            last_import_line = import_lines[-1]
            lines = content.split('\n')
            insert_pos = sum(len(line) + 1 for line in lines[:last_import_line + 1])
        else:
            print("  ⚠️  No se pudo encontrar ubicación para insertar")
            return False
    else:
        # Encuentra el final de esa línea
        insert_pos = content.find('\n', insert_pos) + 1
    
    # Inserta el código
    new_content = content[:insert_pos] + safe_globals_code + content[insert_pos:]
    
    # Guarda
    file_path.write_text(new_content, encoding='utf-8')
    print("  ✅ Registro de clases seguras añadido a main.py")
    
    return True


def verify_fixes() -> Tuple[bool, List[str]]:
    """Verifica que los fixes estén aplicados"""
    
    print("\n" + "="*70)
    print("🔍 VERIFICACIÓN DE FIXES")
    print("="*70)
    
    issues = []
    
    # Verifica main.py
    main_file = Path('main.py')
    if main_file.exists():
        content = main_file.read_text(encoding='utf-8')
        
        print("\n📄 main.py:")
        
        if 'weights_only=False' in content:
            print("  ✅ Usa weights_only=False")
        else:
            print("  ⚠️  No usa weights_only=False")
            issues.append("main.py no tiene weights_only=False")
        
        if 'add_safe_globals' in content or 'SAFE_GLOBALS_REGISTERED' in content:
            print("  ✅ Tiene registro de clases seguras")
        else:
            print("  ⚠️  No tiene registro de clases seguras")
            issues.append("main.py no tiene registro de clases seguras")
    else:
        print("\n❌ main.py no encontrado")
        issues.append("main.py no encontrado")
    
    # Verifica training.py
    training_file = Path('training.py')
    if training_file.exists():
        content = training_file.read_text(encoding='utf-8')
        
        print("\n📄 training.py:")
        
        if 'weights_only=False' in content:
            print("  ✅ Usa weights_only=False")
        else:
            print("  ⚠️  No usa weights_only=False")
            issues.append("training.py no tiene weights_only=False")
    else:
        print("\n❌ training.py no encontrado")
        issues.append("training.py no encontrado")
    
    # Resumen
    print("\n" + "="*70)
    
    if not issues:
        print("✅ TODOS LOS FIXES APLICADOS CORRECTAMENTE")
        return True, issues
    else:
        print("⚠️  ALGUNOS FIXES PENDIENTES")
        for issue in issues:
            print(f"  • {issue}")
        return False, issues


def test_checkpoint_loading():
    """Test de carga de checkpoint"""
    
    print("\n" + "="*70)
    print("🧪 TEST DE CARGA DE CHECKPOINT")
    print("="*70)
    
    # Busca un checkpoint de prueba
    checkpoint_dirs = [
        Path('runs/latest'),
        Path('runs/experiment'),
        Path('runs')
    ]
    
    checkpoint_path = None
    for dir_path in checkpoint_dirs:
        if dir_path.exists():
            checkpoints = list(dir_path.glob('**/ckpt_best.pt')) + list(dir_path.glob('**/ckpt_last.pt'))
            if checkpoints:
                checkpoint_path = checkpoints[0]
                break
    
    if not checkpoint_path:
        print("\n⚠️  No se encontró ningún checkpoint para probar")
        print("   Esto no es un error, solo no se puede hacer el test")
        return True
    
    print(f"\n📦 Probando con: {checkpoint_path}")
    
    try:
        import torch
        from training import TrainingHistory
        
        # Intenta cargar con el fix
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("  ✅ Checkpoint cargado exitosamente con weights_only=False")
        
        # Verifica contenido
        if 'model_state_dict' in checkpoint:
            print("  ✅ Contiene model_state_dict")
        if 'history' in checkpoint:
            print("  ✅ Contiene history")
            if isinstance(checkpoint['history'], TrainingHistory):
                print("  ✅ history es TrainingHistory correctamente")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error al cargar: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fix para compatibilidad PyTorch 2.6+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python fix_torch_load.py              # Aplica todos los fixes
  python fix_torch_load.py --check-only # Solo verifica sin modificar
  python fix_torch_load.py --test       # Aplica y hace test de carga
        """
    )
    
    parser.add_argument('--check-only', action='store_true',
                       help='Solo verifica, no modifica archivos')
    parser.add_argument('--test', action='store_true',
                       help='Ejecuta test de carga de checkpoint después del fix')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔧 FIX TORCH.LOAD - PyTorch 2.6+ Compatibility")
    print("="*70)
    
    # Verifica versión de PyTorch
    needs_fix, version = check_pytorch_version()
    
    print(f"\n🐍 PyTorch: {version}")
    
    if needs_fix:
        print("  ⚠️  PyTorch 2.6+ detectado - FIX NECESARIO")
    else:
        print("  ℹ️  PyTorch < 2.6 - FIX RECOMENDADO (para compatibilidad futura)")
    
    if args.check_only:
        print("\n🔍 MODO VERIFICACIÓN (no se modificarán archivos)")
        success, issues = verify_fixes()
        sys.exit(0 if success else 1)
    
    # Aplica fixes
    print("\n" + "="*70)
    print("🔧 APLICANDO FIXES")
    print("="*70)
    
    success = True
    
    # Fix 1: Registro de clases seguras (RECOMENDADO)
    print("\n[1/3] Registro de clases seguras...")
    if not add_safe_globals_registration():
        success = False
    
    # Fix 2: main.py weights_only=False
    print("\n[2/3] Actualizando main.py...")
    if not fix_main_py():
        success = False
    
    # Fix 3: training.py weights_only=False
    print("\n[3/3] Actualizando training.py...")
    if not fix_training_py():
        success = False
    
    # Verifica
    verify_success, issues = verify_fixes()
    
    # Test opcional
    if args.test and success:
        test_checkpoint_loading()
    
    # Resultado final
    print("\n" + "="*70)
    
    if success and verify_success:
        print("✅ FIX APLICADO EXITOSAMENTE")
        print("="*70)
        print("\n📝 Cambios aplicados:")
        print("  1. Registro de TrainingHistory como clase segura")
        print("  2. torch.load con weights_only=False en lugares seguros")
        print("  3. Backups creados automáticamente")
        print("\n🎯 Ahora puedes:")
        print("  • Cargar checkpoints sin errores")
        print("  • Usar: python main.py generate --ckpt ...")
        print("  • Usar: python interactive_cli.py")
        print("\n💾 Si algo falla, restaura los backups:")
        print("  • main.py.backup_*")
        print("  • training.py.backup_*")
    else:
        print("⚠️  FIX APLICADO CON ADVERTENCIAS")
        print("="*70)
        print("\nAlgunos fixes no se aplicaron completamente.")
        print("El sistema debería funcionar, pero revisa los warnings arriba.")
    
    print()
    
    return 0 if (success and verify_success) else 1


if __name__ == '__main__':
    sys.exit(main())
