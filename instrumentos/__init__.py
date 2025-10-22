# instrumentos/__init__.py - CORREÇÃO

import os
import importlib
import logging

logger = logging.getLogger('instrumentos')

# Importar os módulos existentes explicitamente
try:
    from . import flauta
    _available_instruments = {'flauta': flauta}
except ImportError as e:
    logger.error(f"Erro ao importar módulo flauta: {e}")
    _available_instruments = {}

# Detectar automaticamente outros módulos de instrumentos
_instruments_dir = os.path.dirname(__file__)
for filename in os.listdir(_instruments_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Remover .py
        if module_name not in _available_instruments:
            try:
                module = importlib.import_module(f'.{module_name}', package=__name__)
                _available_instruments[module_name] = module
                logger.info(f"Módulo de instrumento carregado: {module_name}")
            except ImportError as e:
                logger.warning(f"Não foi possível importar o módulo {module_name}: {e}")

# Lista de instrumentos disponíveis
available_instruments = list(_available_instruments.keys())

def get_instrument_module(instrument_name):
    """
    Retorna o módulo do instrumento especificado.
    
    Args:
        instrument_name (str): Nome do instrumento
        
    Returns:
        module: Módulo do instrumento
        
    Raises:
        ImportError: Se o módulo não for encontrado
    """
    # Normalizar o nome do instrumento para lowercase
    instrument_name = instrument_name.lower()
    
    if instrument_name in _available_instruments:
        return _available_instruments[instrument_name]
    else:
        # Tentar carregar o módulo se não estiver no cache
        try:
            module = importlib.import_module(f'.{instrument_name}', package=__name__)
            _available_instruments[instrument_name] = module
            return module
        except ImportError:
            raise ImportError(f"Instrumento '{instrument_name}' não encontrado. Instrumentos disponíveis: {', '.join(available_instruments)}")
