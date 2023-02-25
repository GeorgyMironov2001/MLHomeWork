import numpy as np
from src.nn.module.parameters import Parameters


class ReLU:
    """Реализует ReLU"""

    def __init__(self):
        self.params = Parameters(1)
        self.input = None
        self.out = None

    def forward(self, inpt):
        """Реализует forward-pass

        ---------
        Параметры
        ---------
        inpt : np.ndarray, shape=(M, N_in)
            Входные данные

        ----------
        Возвращает
        ----------
        output : np.ndarray, shape=(M, N_in)
            Выход слоя
        """
        # TODO: Реализовать рассчет RELU функции активации
        self.input = inpt
        self.out = np.maximum(0, inpt)
        return self.out

    def __call__(self, *inpt):
        """Аналогично forward"""
        return self.forward(*inpt)

    def parameters(self):
        """Возвращает параметры модели"""
        return self.params

    def _zero_grad(self):
        """Обнуляет градиенты модели

        Не нужен в данном случае,
        оставим для совместимости
        """
        pass

    def _compute_gradients(self, grads):
        """Считает градиенты модели"""
        # TODO: Реализовать рассчет градиентов
        input_grads = np.multiply(grads, self.input > 0)
        return input_grads

    def _train(self):
        """Переводит модель в режим обучения"""
        pass

    def _eval(self):
        """Переводит модель в режим оценивания"""
        pass

    def __repr__(self):
        return "ReLU()"
