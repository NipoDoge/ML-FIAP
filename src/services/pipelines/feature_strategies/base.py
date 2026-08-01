import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStrategy(ABC):
    """
    Contrato base para strategies de feature engineering.
    Cada domínio (cardiologia, churn, etc.) implementa sua própria strategy.
    """

    @abstractmethod
    def required_columns(self) -> list[str]:
        """Colunas mínimas que o dataset deve ter para esta strategy funcionar."""
        ...

    @abstractmethod
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recebe o DataFrame pré-processado e retorna com as novas features adicionadas.
        Não deve modificar o DataFrame original (trabalhar sobre cópia).
        """
        ...

    @abstractmethod
    def created_features(self) -> list[str]:
        """Nomes das features que esta strategy pode criar."""
        ...

    def validate(self, df: pd.DataFrame) -> None:
        """
        Verifica se as colunas obrigatórias estão presentes.

        Usa apenas nomes normalizados (``strip()`` + ``lower()``) para a comparação, **sem
        renomear** colunas nem alterar ``df``. Assim ficheiros com ``MonthlyCharges`` passam ao
        mesmo tempo que o baseline ou outro código continuam a ver os cabeçalhos originais.
        """
        normalized_headers = {str(c).strip().lower() for c in df.columns}
        missing = [col for col in self.required_columns() if col not in normalized_headers]
        if missing:
            logger.error(f"Colunas obrigatórias ausentes para {self.__class__.__name__}: {missing}")
            raise ValueError(
                f"Dataset incompatível com {self.__class__.__name__}. Colunas ausentes: {missing}"
            )
        logger.info(
            f"Validação OK — todas as colunas obrigatórias presentes para {self.__class__.__name__}"
        )
