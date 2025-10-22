# setup_integration.py
"""
Script para instalar e integrar todas as novas funcionalidades
no sistema de análise de densidade musical existente.
"""

import os
import shutil
import importlib
import sys
import logging
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('integration_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('setup_integration')

def criar_estrutura_diretorios():
    """Cria a estrutura de diretórios necessária se não existir."""
    diretorios = [
        'calibration',
        'config',
        'instrumentos',
    ]
    
    for diretorio in diretorios:
        if not os.path.exists(diretorio):
            os.makedirs(diretorio, exist_ok=True)
            logger.info(f"Diretório criado: {diretorio}")
    
    # Diretório config dentro de calibration
    if not os.path.exists(os.path.join('calibration', 'config')):
        os.makedirs(os.path.join('calibration', 'config'), exist_ok=True)
        logger.info(f"Diretório criado: calibration/config")


def mover_arquivos_novos():
    """Move os novos arquivos para os diretórios corretos."""
    # Lista de arquivos e seus destinos
    arquivos = [
        ('calibration.py', ''),  # Raiz
        ('microtonal_utils.py', ''),  # Raiz
        ('gui_calibration.py', ''),  # Raiz
        ('microtonal_gui.py', ''),  # Raiz
        ('clarinete.py', 'instrumentos/'),  # Pasta de instrumentos
    ]
    
    for arquivo, destino in arquivos:
        origem = arquivo  # Arquivo no diretório atual
        caminho_destino = os.path.join(destino, arquivo)
        
        if os.path.exists(origem):
            # Criar backup se o arquivo já existir no destino
            if os.path.exists(caminho_destino):
                backup_path = f"{caminho_destino}.bak"
                shutil.copy2(caminho_destino, backup_path)
                logger.info(f"Backup criado: {backup_path}")
            
            # Copiar arquivo
            shutil.copy2(origem, caminho_destino)
            logger.info(f"Arquivo copiado: {origem} -> {caminho_destino}")
        else:
            logger.warning(f"Arquivo de origem não encontrado: {origem}")


def criar_arquivo_inicializacao():
    """Cria arquivo __init__.py na pasta calibration se não existir."""
    init_path = os.path.join('calibration', '__init__.py')
    
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            f.write('# calibration/__init__.py\n')
            f.write('# Pacote para funções de calibração do parâmetro lambda\n\n')
            f.write('from .calibration import *\n')
        
        logger.info(f"Arquivo criado: {init_path}")


def criar_arquivo_configuracao():
    """Cria arquivo de configuração padrão com lambda inicial."""
    import json
    
    config_path = os.path.join('calibration', 'config', 'density_params.json')
    
    if not os.path.exists(config_path):
        config_data = {
            'lambda': 0.05  # Valor padrão inicial
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        logger.info(f"Arquivo de configuração criado: {config_path}")


def modificar_main_py():
    """Modifica o arquivo Main.py para integrar as novas funcionalidades."""
    main_path = 'Main.py'
    
    if not os.path.exists(main_path):
        logger.error(f"Arquivo Main.py não encontrado")
        return
    
    # Ler o conteúdo atual
    with open(main_path, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    # Criar backup
    backup_path = f"{main_path}.bak"
    shutil.copy2(main_path, backup_path)
    logger.info(f"Backup do Main.py criado: {backup_path}")
    
    # Verificar se já tem as importações
    if 'from gui_calibration import' not in conteudo:
        # Encontrar a seção de importações
        import_section = conteudo.find('import')
        if import_section >= 0:
            import_end = conteudo.find('\n\n', import_section)
            if import_end < 0:
                import_end = len(conteudo)
            
            # Novas importações
            novos_imports = (
                "\n# Importar módulos para calibração e GUI avançada\n"
                "from gui_calibration import adicionar_menu_calibracao, abrir_janela_calibracao\n"
                "from microtonal_gui import integrate_with_density_calculator\n"
                "from calibration import obter_lambda_atual\n"
            )
            
            # Inserir novas importações
            conteudo_modificado = (
                conteudo[:import_end] + 
                novos_imports + 
                conteudo[import_end:]
            )
            
            # Encontrar a classe DensityAnalyzerApp
            app_class = conteudo_modificado.find('class DensityAnalyzerApp')
            if app_class >= 0:
                # Encontrar o método __init__
                init_method = conteudo_modificado.find('def __init__', app_class)
                if init_method >= 0:
                    # Encontrar o final da inicialização
                    init_end = conteudo_modificado.find('\n\n', init_method)
                    if init_end > 0:
                        # Código para adicionar menu de calibração
                        menu_code = (
                            "\n        # Adicionar menu de calibração\n"
                            "        self._adicionar_menu_calibracao()\n"
                            "\n        # Integrar widgets microtonais\n"
                            "        self.microtonal_selector = integrate_with_density_calculator(self.gui)\n"
                        )
                        
                        conteudo_modificado = (
                            conteudo_modificado[:init_end] + 
                            menu_code + 
                            conteudo_modificado[init_end:]
                        )
                        
                        # Adicionar método para menu de calibração
                        method_add = (
                            "\n    def _adicionar_menu_calibracao(self):\n"
                            "        \"\"\"Adiciona um menu para as funções de calibração.\"\"\"\n"
                            "        try:\n"
                            "            # Obter menubar existente ou criar novo\n"
                            "            menubar = None\n"
                            "            if hasattr(self.root, 'config') and callable(self.root.config):\n"
                            "                menubar_name = self.root.cget('menu')\n"
                            "                if menubar_name:\n"
                            "                    menubar = self.root.nametowidget(menubar_name)\n"
                            "\n"
                            "            # Adicionar opções de calibração\n"
                            "            adicionar_menu_calibracao(self.root, menubar, self.calibrar_lambda_callback)\n"
                            "        except Exception as e:\n"
                            "            logger.error(f\"Erro ao adicionar menu de calibração: {e}\")\n"
                            "\n"
                            "    def calibrar_lambda_callback(self, novo_lambda):\n"
                            "        \"\"\"Callback chamado quando a calibração é concluída.\"\"\"\n"
                            "        logger.info(f\"Lambda recalibrado para: {novo_lambda}\")\n"
                            "        # Atualizar qualquer componente que dependa de lambda, se necessário\n"
                        )
                        
                        # Encontrar o fim da classe
                        class_end = conteudo_modificado.find('# Ponto de entrada principal')
                        if class_end > 0:
                            conteudo_modificado = (
                                conteudo_modificado[:class_end] + 
                                method_add + 
                                conteudo_modificado[class_end:]
                            )
                
            # Salvar o arquivo modificado
            with open(main_path, 'w', encoding='utf-8') as f:
                f.write(conteudo_modificado)
            
            logger.info(f"Arquivo Main.py modificado com sucesso")
        else:
            logger.error(f"Não foi possível encontrar a seção de importações em Main.py")
    else:
        logger.info(f"Arquivo Main.py já contém as modificações necessárias")


def verificar_dependencias():
    """Verifica se todas as dependências necessárias estão instaladas."""
    dependencias = [
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'sklearn',
    ]
    
    faltantes = []
    for dep in dependencias:
        try:
            importlib.import_module(dep)
        except ImportError:
            faltantes.append(dep)
    
    if faltantes:
        logger.warning(f"Dependências faltantes: {', '.join(faltantes)}")
        print("\nAlgumas dependências estão faltando. Execute o comando:")
        print(f"pip install {' '.join(faltantes)}")
    else:
        logger.info("Todas as dependências estão instaladas")


def executar_integracao():
    """Executa todo o processo de integração."""
    try:
        print("Iniciando processo de integração...")
        logger.info("Iniciando processo de integração")
        
        # Verificar dependências
        verificar_dependencias()
        
        # Criar estrutura de diretórios
        criar_estrutura_diretorios()
        
        # Criar arquivo __init__.py
        criar_arquivo_inicializacao()
        
        # Criar arquivo de configuração
        criar_arquivo_configuracao()
        
        # Mover arquivos novos
        mover_arquivos_novos()
        
        # Modificar Main.py
        modificar_main_py()
        
        print("\nIntegração concluída com sucesso!")
        logger.info("Integração concluída com sucesso")
        
        print("\nPróximos passos:")
        print("1. Verifique o arquivo 'integration_log.txt' para detalhes da integração")
        print("2. Execute a aplicação normalmente: python Main.py")
        print("3. Acesse as novas funcionalidades pelo menu 'Ferramentas'")
        
    except Exception as e:
        logger.error(f"Erro durante a integração: {e}")
        logger.error(traceback.format_exc())
        print(f"\nErro durante a integração: {e}")
        print("Verifique o arquivo 'integration_log.txt' para mais detalhes")


if __name__ == "__main__":
    executar_integracao()
