# music_widget.py - versão simplificada e robusta

# Widget personalizado para exibição de notação musical em Tkinter



import tkinter as tk

from tkinter import ttk, font

#from duration_utils import DuracaoMusical



class MusicalSymbolLabel(tk.Label):

    """

    Um widget de etiqueta que exibe símbolos musicais.

    Usa abreviações de texto já que fontes musicais podem não estar disponíveis.

    """

    

    def __init__(self, master, **kwargs):

        """

        Inicializa o widget.

        

        Args:

            master: Widget pai

            duration_value (float): Valor da duração musical

            **kwargs: Argumentos adicionais para o Label

        """

        super().__init__(master, **kwargs)

        self.duration_value = duration_value

        self.update_symbol()

    

    def update_symbol(self):

        """

        Atualiza o símbolo exibido baseado no valor da duração.

       # """

        nome = DuracaoMusical.valor_para_nome(self.duration_value)

        simbolo_texto = DuracaoMusical.SIMBOLOS_TEXTO.get(nome, "")

        self.config(text=f"{simbolo_texto}", font=("Arial", 12, "bold"))





class DurationSelector(ttk.Frame):

    """

    Widget composto para seleção de duração com visualização do símbolo musical.

    """

    

    def __init__(self, master, initial_value=4, **kwargs):

        """

        Inicializa o seletor de duração.

        

        Args:

            master: Widget pai

            initial_value (float): Valor inicial da duração

            **kwargs: Argumentos adicionais para o Frame

        """

        # Extrair parâmetro 'state' se existir

        self._state = kwargs.pop('state', 'normal')

        

        super().__init__(master, **kwargs)

        

        self.duration_var = tk.StringVar()

        

        # Obter opções de duração

        #self.duration_options = DuracaoMusical.get_tkinter_duration_options()

        

        # Definir valor inicial

        #initial_name = DuracaoMusical.valor_para_nome(initial_value)

        #for option in self.duration_options:

         #   if option.startswith(initial_name):

          #      self.duration_var.set(option)

           #     break

        

        # Criar combobox para seleção

        self.duration_menu = ttk.Combobox(

            self, textvariable=self.duration_var,

            values=self.duration_options, width=15,

            state=self._state

        )

        self.duration_menu.grid(row=0, column=0, padx=5, pady=2)

        

        # Criar widget de símbolo musical

        self.symbol_widget = MusicalSymbolLabel(

            self, duration_value=initial_value,

            width=3, relief="ridge", padx=5

        )

        self.symbol_widget.grid(row=0, column=1, padx=5, pady=2)

        

        # Configurar callback para atualização

        self.duration_var.trace_add("write", self._update_symbol)

    

    #def _update_symbol(self, *args):

        """Atualiza o símbolo quando a seleção muda."""

        #selected = self.duration_var.get()

        #duration_value = DuracaoMusical.parse_duracao_string(selected)

        #self.symbol_widget.duration_value = duration_value

        #self.symbol_widget.update_symbol()

    

    def get(self):

        """Retorna a string selecionada."""

        return self.duration_var.get()

    

    #def get_value(self):

      #  """Retorna o valor numérico da duração selecionada."""

     #   return DuracaoMusical.parse_duracao_string(self.duration_var.get())

    

    def configure(self, **kwargs):

        """

        Configura o widget e seus componentes.

        

        Args:

            **kwargs: Argumentos de configuração

        """

        # Verificar se temos uma mudança de estado

        if 'state' in kwargs:

            self._state = kwargs['state']

            self.duration_menu.configure(state=self._state)

            # Remover para não causar erro no Frame

            kwargs.pop('state')

            

        # Passar os outros argumentos para o método pai

        super().configure(**kwargs)

    

    # Alias para configure para compatibilidade com API Tkinter

    config = configure





# Função de demonstração

def main():

    """Demonstração do widget."""

    root = tk.Tk()

    root.title("Seletor de Duração Musical")

    

    # Criar widget

    selector = DurationSelector(root, initial_value=4)

    selector.pack(padx=20, pady=20)

    

    # Botão para mostrar valor selecionado

    def show_value():

        print(f"String selecionada: {selector.get()}")

        print(f"Valor numérico: {selector.get_value()}")

    

    tk.Button(root, text="Mostrar Valor", command=show_value).pack(pady=10)

    

    root.mainloop()



if __name__ == "__main__":

    main()

