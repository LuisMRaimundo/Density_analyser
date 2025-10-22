# error_handler.py

# Módulo para tratamento centralizado de exceções



import logging

import traceback

import sys

from typing import Optional, Any, Callable, TypeVar, Dict, List, Union

from functools import wraps

from tkinter import messagebox



# Importar mensagens de erro da configuração

from config import ERROR_MESSAGES



# Configurar logger

logger = logging.getLogger('error_handler')



T = TypeVar('T')  # Define type variable for return types



# Define custom exception classes for the application

class DensidadeError(Exception):

    """Base exception class for the application."""

    def __init__(self, message: str = "Erro na aplicação", code: str = "ERR_GENERIC"):

        self.message = message

        self.code = code

        super().__init__(self.message)





class InputError(DensidadeError):

    """Exception for input data errors."""

    def __init__(self, message: str = "Erro nos dados de entrada", field: Optional[str] = None):

        code = "ERR_INPUT"

        self.field = field

        if field:

            message = f"{message} (campo: {field})"

        super().__init__(message, code)





class CalculationError(DensidadeError):

    """Exception for calculation errors."""

    def __init__(self, message: str = "Erro durante cálculos", details: Optional[str] = None):

        code = "ERR_CALCULATION"

        self.details = details

        if details:

            message = f"{message}: {details}"

        super().__init__(message, code)





class FileOperationError(DensidadeError):

    """Exception for file operation errors."""

    def __init__(self, message: str = "Erro em operação de arquivo", 

                 filename: Optional[str] = None, operation: Optional[str] = None):

        code = "ERR_FILE"

        self.filename = filename

        self.operation = operation

        

        msg_parts = [message]

        if operation:

            msg_parts.append(f"operação: {operation}")

        if filename:

            msg_parts.append(f"arquivo: {filename}")

            

        super().__init__(", ".join(msg_parts), code)





class ModuleError(DensidadeError):

    """Exception for module loading and execution errors."""

    def __init__(self, message: str = "Erro em módulo", module_name: Optional[str] = None):

        code = "ERR_MODULE"

        self.module_name = module_name

        if module_name:

            message = f"{message} (módulo: {module_name})"

        super().__init__(message, code)





# Decorator for exception handling in functions

def handle_exceptions(show_dialog: bool = True, 
                     fallback_value: Any = None, 
                     rethrow: bool = False) -> Callable:
    """
    Decorator to handle exceptions in a standardized way.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except DensidadeError as e:
                logger.error(f"{e.code}: {e.message}")
                if show_dialog:
                    messagebox.showerror("Erro", e.message)
                if rethrow:
                    raise
                return fallback_value
            except Exception as e:
                error_msg = str(e)
                stack_trace = traceback.format_exc()
                logger.error(f"Unexpected error: {error_msg}\n{stack_trace}")
                if show_dialog:
                    messagebox.showerror("Erro inesperado", 
                                       f"{error_msg}\n\nDetalhes técnicos foram registrados no log.")
                if rethrow:
                    raise
                return fallback_value
        return wrapper
    return decorator


def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Union[T, Any]:
        try:
            return func(*args, **kwargs)
        except DensidadeError as e:
            # Handle our custom exceptions
            logger.error(f"{e.code}: {e.message}")
            if show_dialog:
                messagebox.showerror("Erro", e.message)
            if rethrow:
                raise
            return fallback_value
        except Exception as e:
            # Handle unexpected exceptions
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            logger.error(f"Unexpected error: {error_msg}\n{stack_trace}")
            
            if show_dialog:
                messagebox.showerror("Erro inesperado", 
                                   f"{error_msg}\n\nDetalhes técnicos foram registrados no log.")
            if rethrow:
                raise
            return fallback_value
            
    return wrapper





# Function to log and display errors

def log_and_show_error(error: Exception, show_dialog: bool = True, 

                      title: str = "Erro", additional_info: Optional[str] = None) -> None:

    """

    Logs an error and optionally shows it in a dialog.

    

    Args:

        error (Exception): The exception to log and show

        show_dialog (bool): Whether to show a message dialog

        title (str): Title for the error dialog

        additional_info (str, optional): Additional information to include

    """

    error_msg = str(error)

    stack_trace = traceback.format_exc()

    

    # Format the message

    if additional_info:

        log_msg = f"{error_msg} - {additional_info}\n{stack_trace}"

        display_msg = f"{error_msg}\n\n{additional_info}"

    else:

        log_msg = f"{error_msg}\n{stack_trace}"

        display_msg = error_msg

    

    # Log the error

    logger.error(log_msg)

    

    # Show dialog if requested

    if show_dialog:

        messagebox.showerror(title, display_msg)





# Function to map exception to error message from config

def get_error_message(error_type: str, **kwargs) -> str:

    """

    Gets a formatted error message from the config.

    

    Args:

        error_type (str): The type of error to get a message for

        **kwargs: Format parameters for the message

        

    Returns:

        str: Formatted error message

    """

    message = ERROR_MESSAGES.get(error_type, "Erro desconhecido")

    

    try:

        return message.format(**kwargs)

    except KeyError:

        # If formatting fails, return the original message

        logger.warning(f"Could not format error message: {message} with params {kwargs}")

        return message





# Context manager for handling exceptions

class ErrorContext:

    """

    Context manager for handling exceptions in a standardized way.

    

    Usage:

        with ErrorContext("Operação de arquivo", show_dialog=True):

            # code that might raise exceptions

    """

    def __init__(self, operation_name: str, show_dialog: bool = True, 

                rethrow: bool = False, fallback_value: Any = None):

        """

        Initialize the context manager.

        

        Args:

            operation_name (str): Name of the operation for logging

            show_dialog (bool): Whether to show a dialog on error

            rethrow (bool): Whether to rethrow the exception

            fallback_value: Value to return if an exception occurs

        """

        self.operation_name = operation_name

        self.show_dialog = show_dialog

        self.rethrow = rethrow

        self.fallback_value = fallback_value

        self.result = None

    

    def __enter__(self):

        return self

    

    def __exit__(self, exc_type, exc_value, exc_traceback):

        if exc_type is not None:

            # An exception occurred

            error_msg = f"Erro durante: {self.operation_name}"

            if exc_value:

                error_msg += f" - {str(exc_value)}"

            

            stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

            logger.error(f"{error_msg}\n{stack_trace}")

            

            if self.show_dialog:

                messagebox.showerror("Erro", error_msg)

            

            self.result = self.fallback_value

            return not self.rethrow  # Suppress exception if not rethrowing

        

        return True  # No exception occurred





# Initialize global exception hook

def init_global_exception_hook():

    """

    Sets up a global exception hook to catch uncaught exceptions.

    """

    def global_exception_handler(exc_type, exc_value, exc_traceback):

        # Skip KeyboardInterrupt

        if issubclass(exc_type, KeyboardInterrupt):

            sys.__excepthook__(exc_type, exc_value, exc_traceback)

            return

        

        # Format the error

        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        

        # Log it

        logger.critical(f"Uncaught exception: {error_msg}")

        

        # Show a dialog in Tkinter if possible

        try:

            messagebox.showerror("Erro crítico", 

                               "Ocorreu um erro inesperado e a aplicação pode estar instável.\n"

                               "Recomenda-se salvar seu trabalho (se possível) e reiniciar a aplicação.\n\n"

                               f"Erro: {str(exc_value)}")

        except:

            # If showing dialog fails, just print

            print(f"ERRO CRÍTICO: {str(exc_value)}")

            print(error_msg)

    

    # Set the hook

    sys.excepthook = global_exception_handler





# Validate input data with clear error messages

def validate_input_data(data: Dict[str, Any], required_fields: List[str]) -> None:

    """

    Validates that required fields are present and have valid values.

    

    Args:

        data (dict): Data to validate

        required_fields (list): List of required field names

        

    Raises:

        InputError: If validation fails

    """

    for field in required_fields:

        if field not in data:

            raise InputError(f"Campo obrigatório ausente: {field}", field)

        

        value = data[field]

        if value is None:

            raise InputError(f"Valor não pode ser nulo", field)

        

        if isinstance(value, list) and len(value) == 0:

            raise InputError(f"Lista não pode estar vazia", field)

        

        if isinstance(value, str) and value.strip() == "":

            raise InputError(f"String não pode estar vazia", field)

