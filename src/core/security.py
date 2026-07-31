from passlib.context import CryptContext

CRYPT = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(password: str, hash_password: str) -> bool:
    return CRYPT.verify(password, hash_password)

    """
        Essa função vai receber a senha em texto informado pelo usuario e 
        o retorno do hash vindo do banco de dados, e vai fazer a comparação
        a função verify, vai transformar a senha em hash e comparar com o hash já salvo.
    """


def get_password_hash(password: str) -> str:
    return CRYPT.hash(password)
    """
        Essa função vai gerar um hash da senha que será salva.
    
    """
