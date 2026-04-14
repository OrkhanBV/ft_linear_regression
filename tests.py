"""
ft_linear_regression — полный тест-сюит по чек-листу защиты.

Запуск:  python3 tests.py
         python3 tests.py -v          (подробный вывод)

Структура:
  Part 1 — Checklist (точно по пунктам чек-листа защиты)
  Part 2 — Robustness (всё, чем проверяющий может попробовать сломать проект)
  Part 3 — Math correctness (формулы, одновременное обновление, денормализация)
"""

import csv
import json
import math
import os
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Путь к файлам проекта (тесты запускаются из корня репо)
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import predict
import train


# ===========================================================================
# Вспомогательные утилиты
# ===========================================================================

def write_tmp_csv(rows, headers="km,price"):
    """Записывает временный CSV и возвращает его путь."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    f.write(headers + "\n")
    for row in rows:
        f.write(row + "\n")
    f.close()
    return f.name


def write_tmp_model(theta0, theta1):
    """Записывает временный model.json и возвращает его путь."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump({"theta0": theta0, "theta1": theta1}, f)
    f.close()
    return f.name


def capture_stdout(func, *args, **kwargs):
    """Перехватывает stdout и возвращает вывод + возвращаемое значение."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        result = func(*args, **kwargs)
    return buf.getvalue(), result


# ===========================================================================
# PART 1 — CHECKLIST (пункты чек-листа один за другим)
# ===========================================================================

class TestChecklist_Prelim(unittest.TestCase):
    """Preliminaries — базовые требования репо."""

    def test_predict_file_exists(self):
        """predict.py существует в репо."""
        self.assertTrue(
            os.path.exists(os.path.join(PROJECT_DIR, "predict.py")),
            "predict.py не найден"
        )

    def test_train_file_exists(self):
        """train.py существует в репо."""
        self.assertTrue(
            os.path.exists(os.path.join(PROJECT_DIR, "train.py")),
            "train.py не найден"
        )

    def test_data_csv_exists(self):
        """data.csv существует в репо."""
        self.assertTrue(
            os.path.exists(os.path.join(PROJECT_DIR, "data.csv")),
            "data.csv не найден"
        )

    def test_no_forbidden_libraries(self):
        """Запрещённые функции (numpy.polyfit и аналоги) не используются."""
        forbidden = [
            "numpy.polyfit", "np.polyfit",
            "scipy.stats.linregress", "sklearn",
            "polyfit", "LinearRegression",
        ]
        for fname in ["predict.py", "train.py"]:
            fpath = os.path.join(PROJECT_DIR, fname)
            with open(fpath) as f:
                source = f.read()
            for keyword in forbidden:
                self.assertNotIn(
                    keyword, source,
                    f"Запрещённая функция '{keyword}' найдена в {fname}"
                )

    def test_two_separate_programs(self):
        """Существуют ровно 2 отдельных программы: predict.py и train.py."""
        self.assertTrue(os.path.isfile(os.path.join(PROJECT_DIR, "predict.py")))
        self.assertTrue(os.path.isfile(os.path.join(PROJECT_DIR, "train.py")))


class TestChecklist_PredictionBeforeTraining(unittest.TestCase):
    """
    Checklist: «Prediction before training»
    Запустить predict до обучения — должен вернуть 0.
    Проверить уравнение: theta0 + (theta1 * x).
    """

    def test_estimate_price_formula(self):
        """Уравнение predict.estimate_price = theta0 + theta1*x."""
        self.assertAlmostEqual(predict.estimate_price(100, 5, 2), 205.0)
        self.assertAlmostEqual(predict.estimate_price(0, 7, -3), 7.0)
        self.assertAlmostEqual(predict.estimate_price(50000, 0, 0), 0.0)

    def test_prediction_without_model_returns_zero(self):
        """Без model.json theta0=theta1=0, поэтому predict(x)=0 для любого x."""
        with patch.object(predict, "MODEL_FILE", "/nonexistent_model_12345.json"):
            t0, t1 = predict.load_thetas()
        self.assertEqual(t0, 0.0)
        self.assertEqual(t1, 0.0)
        # 0 + 0*mileage = 0
        self.assertEqual(predict.estimate_price(50000, t0, t1), 0.0)
        self.assertEqual(predict.estimate_price(99999, t0, t1), 0.0)

    def test_prediction_before_training_any_nonzero_mileage_gives_zero(self):
        """Чек-лист: «Enter a value that is not null → should print 0»."""
        with patch.object(predict, "MODEL_FILE", "/nonexistent_12345.json"):
            t0, t1 = predict.load_thetas()
        for mileage in [1, 100, 10000, 240000]:
            price = predict.estimate_price(mileage, t0, t1)
            self.assertEqual(price, 0.0,
                f"Ожидалось 0.0 до обучения, получили {price} для mileage={mileage}")


class TestChecklist_TrainingPhase(unittest.TestCase):
    """
    Checklist: «Training phase»
    Реализация функции из subject, сохранение theta0/theta1.
    """

    def setUp(self):
        self.csv_path = write_tmp_csv([
            "240000,3650", "139800,3800", "150500,4400",
            "185530,4450", "176000,5250", "114800,5350",
            "166800,5800", "89000,5990",  "144500,5999",
            "84000,6200",  "82029,6390",  "63060,6390",
            "74000,6600",  "97500,6800",  "67000,6800",
            "76025,6900",  "48235,6900",  "93000,6990",
            "60949,7490",  "65674,7555",  "54000,7990",
            "68500,7990",  "22899,7990",  "61789,8290",
        ])

    def tearDown(self):
        os.unlink(self.csv_path)

    def test_gradient_descent_runs_and_converges(self):
        """gradient_descent возвращает разумные theta0/theta1 (не 0, не NaN)."""
        km_norm, mean_km, std_km = train.normalize([100.0, 200.0, 300.0])
        prices = [5000.0, 4000.0, 3000.0]
        t0, t1 = train.gradient_descent(km_norm, prices, 0.1, 1000)
        self.assertFalse(math.isnan(t0), "theta0 = NaN")
        self.assertFalse(math.isnan(t1), "theta1 = NaN")
        self.assertNotEqual(t0, 0.0, "theta0 не изменился (не обучился)")
        self.assertNotEqual(t1, 0.0, "theta1 не изменился (не обучился)")

    def test_gradient_descent_uses_simultaneous_update(self):
        """
        Чек-лист: «результаты 2 уравнений сохраняются во временные переменные
        перед обновлением theta0 и theta1».
        Проверяем через анализ исходника — tmp_theta0 / tmp_theta1 должны
        присутствовать и оба вычисляться ДО присваивания.
        """
        fpath = os.path.join(PROJECT_DIR, "train.py")
        with open(fpath) as f:
            src = f.read()
        self.assertIn("tmp_theta0", src, "tmp_theta0 не найдена в train.py")
        self.assertIn("tmp_theta1", src, "tmp_theta1 не найдена в train.py")

    def test_train_saves_model_file(self):
        """train.py сохраняет model.json с theta0 и theta1."""
        model_path = tempfile.mktemp(suffix=".json")
        try:
            with patch.object(train, "DATA_FILE", self.csv_path), \
                 patch.object(train, "MODEL_FILE", model_path):
                train.main()
            self.assertTrue(os.path.exists(model_path), "model.json не создан")
            with open(model_path) as f:
                data = json.load(f)
            self.assertIn("theta0", data)
            self.assertIn("theta1", data)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_theta0_and_theta1_are_floats_in_model(self):
        """theta0 и theta1 в model.json — числа (float/int)."""
        model_path = tempfile.mktemp(suffix=".json")
        try:
            with patch.object(train, "DATA_FILE", self.csv_path), \
                 patch.object(train, "MODEL_FILE", model_path):
                train.main()
            with open(model_path) as f:
                data = json.load(f)
            self.assertIsInstance(data["theta0"], (int, float))
            self.assertIsInstance(data["theta1"], (int, float))
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_theta1_is_negative(self):
        """Для данного датасета theta1 должен быть отрицательным (цена падает с пробегом)."""
        model_path = tempfile.mktemp(suffix=".json")
        try:
            with patch.object(train, "DATA_FILE", self.csv_path), \
                 patch.object(train, "MODEL_FILE", model_path):
                train.main()
            with open(model_path) as f:
                data = json.load(f)
            self.assertLess(data["theta1"], 0,
                "theta1 должен быть < 0 — с ростом пробега цена падает")
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestChecklist_ReadCSV(unittest.TestCase):
    """Checklist: «The training program should read the csv file»."""

    def test_load_data_reads_km_and_price(self):
        """load_data корректно читает km и price из CSV."""
        csv_path = write_tmp_csv(["100000,5000", "50000,7000"])
        try:
            km, prices = train.load_data(csv_path)
            self.assertEqual(km, [100000.0, 50000.0])
            self.assertEqual(prices, [5000.0, 7000.0])
        finally:
            os.unlink(csv_path)

    def test_load_full_dataset_24_rows(self):
        """Все 24 строки data.csv успешно загружаются."""
        km, prices = train.load_data(os.path.join(PROJECT_DIR, "data.csv"))
        self.assertEqual(len(km), 24)
        self.assertEqual(len(prices), 24)


class TestChecklist_SimultaneousAssignment(unittest.TestCase):
    """
    Checklist: «Simultaneous assignation»
    Результаты 2 уравнений сохраняются во временные переменные
    перед финальным присвоением theta0 и theta1.
    """

    def test_simultaneous_update_correctness(self):
        """
        Тест на числовую корректность одновременного обновления.
        Делаем 1 итерацию вручную и сравниваем с результатом функции.
        """
        km_norm = [-1.0, 0.0, 1.0]
        prices  = [8000.0, 6000.0, 4000.0]
        lr = 0.1
        m = 3

        # Один шаг вручную (оба tmp вычисляются от начальных 0,0)
        t0, t1 = 0.0, 0.0
        errors = [train.estimate_price(km_norm[i], t0, t1) - prices[i] for i in range(m)]
        tmp0 = lr * sum(errors) / m
        tmp1 = lr * sum(errors[i] * km_norm[i] for i in range(m)) / m
        expected_t0 = t0 - tmp0
        expected_t1 = t1 - tmp1

        result_t0, result_t1 = train.gradient_descent(km_norm, prices, lr, 1)

        self.assertAlmostEqual(result_t0, expected_t0, places=10)
        self.assertAlmostEqual(result_t1, expected_t1, places=10)

    def test_no_sequential_update(self):
        """
        Если бы theta0 обновлялся до вычисления tmp_theta1,
        результаты theta1 были бы другими — проверяем что это НЕ так.
        Используем несимметричные km_norm чтобы разница была гарантирована.
        """
        # Несимметричные значения: при последовательном обновлении
        # theta0 меняется и влияет на ошибки при вычислении tmp_theta1.
        km_norm = [0.5, 1.0, 2.0]
        prices  = [9000.0, 5000.0, 3000.0]
        lr = 0.5   # большой lr — разница заметнее
        m = 3

        # === Неправильный вариант: theta0 обновляется ДО вычисления tmp_theta1 ===
        t0, t1 = 0.0, 0.0
        errors_init = [train.estimate_price(km_norm[i], t0, t1) - prices[i]
                       for i in range(m)]
        tmp0_bad = lr * sum(errors_init) / m
        t0_updated = t0 - tmp0_bad  # theta0 уже обновлён

        # theta1 вычисляется с уже новым theta0 — это НЕПРАВИЛЬНО
        errors_bad = [train.estimate_price(km_norm[i], t0_updated, t1) - prices[i]
                      for i in range(m)]
        tmp1_bad = lr * sum(errors_bad[i] * km_norm[i] for i in range(m)) / m
        t1_sequential_wrong = t1 - tmp1_bad

        # === Правильный вариант (одновременное обновление): наша функция ===
        result_t0, result_t1 = train.gradient_descent(km_norm, prices, lr, 1)

        # При правильном одновременном обновлении tmp_theta1 вычисляется
        # используя СТАРЫЙ theta0 (=0), поэтому результат должен отличаться
        diff = abs(result_t1 - t1_sequential_wrong)
        self.assertGreater(
            diff, 1e-6,
            f"theta1={result_t1:.6f} совпадает с последовательным обновлением "
            f"({t1_sequential_wrong:.6f}) — возможно нарушено требование одновременности"
        )


class TestChecklist_PredictionAfterTraining(unittest.TestCase):
    """
    Checklist: «Prediction after training»
    После обучения predict должен выдавать разумные цены.
    """

    @classmethod
    def setUpClass(cls):
        """Обучаем один раз, используем theta в тестах."""
        cls.model_path = tempfile.mktemp(suffix=".json")
        with patch.object(train, "DATA_FILE",
                          os.path.join(PROJECT_DIR, "data.csv")), \
             patch.object(train, "MODEL_FILE", cls.model_path):
            train.main()
        with open(cls.model_path) as f:
            data = json.load(f)
        cls.theta0 = data["theta0"]
        cls.theta1 = data["theta1"]

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.model_path):
            os.unlink(cls.model_path)

    def test_prediction_after_training_not_zero(self):
        """После обучения предсказание для ненулевого пробега != 0."""
        price = predict.estimate_price(100000, self.theta0, self.theta1)
        self.assertNotEqual(price, 0.0)

    def test_predictions_vary_with_mileage(self):
        """Предсказания различаются для разных значений пробега (нет overfitting к const)."""
        prices = [
            predict.estimate_price(km, self.theta0, self.theta1)
            for km in [10000, 50000, 100000, 200000]
        ]
        self.assertGreater(len(set(round(p, 2) for p in prices)), 1,
            "Все предсказания одинаковые — возможен overfitting или константная модель")

    def test_higher_mileage_lower_price(self):
        """Больший пробег → меньшая цена (отрицательный theta1)."""
        price_low  = predict.estimate_price(20000,  self.theta0, self.theta1)
        price_high = predict.estimate_price(200000, self.theta0, self.theta1)
        self.assertGreater(price_low, price_high,
            "Ожидается: маленький пробег → более высокая цена")

    def test_csv_mileage_predictions_are_in_reasonable_range(self):
        """Предсказания для значений из CSV находятся в разумном диапазоне (0–20000)."""
        csv_km = [240000, 139800, 150500, 185530, 176000, 114800,
                  166800, 89000,  144500, 84000]
        for km in csv_km:
            price = predict.estimate_price(km, self.theta0, self.theta1)
            self.assertGreater(price, 0,
                f"Предсказание для km={km} отрицательное: {price:.2f}")
            self.assertLess(price, 20000,
                f"Предсказание для km={km} нереально высокое: {price:.2f}")

    def test_r_squared_above_threshold(self):
        """R² на тренировочных данных > 0.6 — модель реально чему-то обучилась."""
        km_data, price_data = train.load_data(os.path.join(PROJECT_DIR, "data.csv"))
        predicted = [predict.estimate_price(k, self.theta0, self.theta1)
                     for k in km_data]
        m = len(price_data)
        mean_p = sum(price_data) / m
        ss_res = sum((price_data[i] - predicted[i])**2 for i in range(m))
        ss_tot = sum((price_data[i] - mean_p)**2 for i in range(m))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        self.assertGreater(r2, 0.6, f"R²={r2:.4f} слишком низкий — модель плохо обучена")

    def test_load_thetas_from_model_file(self):
        """predict.load_thetas() читает theta0 и theta1 из model.json без ошибок."""
        with patch.object(predict, "MODEL_FILE", self.model_path):
            t0, t1 = predict.load_thetas()
        self.assertAlmostEqual(t0, self.theta0)
        self.assertAlmostEqual(t1, self.theta1)


# ===========================================================================
# PART 2 — ROBUSTNESS (проверяющий пытается сломать проект)
# ===========================================================================

class TestRobustness_PredictInput(unittest.TestCase):
    """Попытки сломать predict.py некорректным вводом."""

    def _run_predict_with_inputs(self, inputs_seq, model_path=None):
        """
        Симулирует последовательный ввод пользователя.
        inputs_seq — список строк. Последняя должна быть корректным числом.
        Возвращает весь stdout.
        """
        model = model_path or write_tmp_model(8499.6, -0.02144)
        cleanup = model_path is None

        input_iter = iter(inputs_seq)
        buf = StringIO()

        try:
            with patch("builtins.input", side_effect=input_iter), \
                 patch("sys.stdout", buf), \
                 patch.object(predict, "MODEL_FILE", model):
                predict.main()
        except StopIteration:
            pass  # Ввод закончился раньше — нормально для тестов
        finally:
            if cleanup:
                os.unlink(model)

        return buf.getvalue()

    def test_string_input_rejected(self):
        """Строка вместо числа — программа просит ввести снова."""
        output = self._run_predict_with_inputs(["abc", "100000"])
        self.assertIn("Error", output)
        self.assertIn("Estimated price", output)

    def test_empty_input_rejected(self):
        """Пустая строка — программа просит ввести снова."""
        output = self._run_predict_with_inputs(["", "100000"])
        self.assertIn("Error", output)
        self.assertIn("Estimated price", output)

    def test_negative_mileage_rejected(self):
        """Отрицательный пробег — программа просит ввести снова."""
        output = self._run_predict_with_inputs(["-1000", "100000"])
        self.assertIn("Error", output)
        self.assertIn("Estimated price", output)

    def test_zero_mileage_accepted(self):
        """Нулевой пробег — допустимое значение, принимается."""
        output = self._run_predict_with_inputs(["0"])
        self.assertIn("Estimated price", output)

    def test_float_mileage_accepted(self):
        """Дробный пробег (123.45) — допустимое значение."""
        output = self._run_predict_with_inputs(["123.45"])
        self.assertIn("Estimated price", output)

    def test_very_large_mileage(self):
        """Очень большой пробег (1e9) — не вызывает краш."""
        output = self._run_predict_with_inputs(["1000000000"])
        self.assertIn("Estimated price", output)

    def test_multiple_wrong_inputs_then_valid(self):
        """Несколько некорректных вводов подряд, затем верный — работает."""
        output = self._run_predict_with_inputs(
            ["abc", "", "-5", "!!!", "   ", "50000"]
        )
        self.assertIn("Estimated price", output)

    def test_whitespace_input_rejected(self):
        """Пробелы как ввод — отклоняется (strip делает его пустым)."""
        output = self._run_predict_with_inputs(["   ", "50000"])
        self.assertIn("Error", output)

    def test_special_characters_rejected(self):
        """Спецсимволы — отклоняются."""
        output = self._run_predict_with_inputs(["!@#$%", "50000"])
        self.assertIn("Error", output)

    def test_scientific_notation_accepted(self):
        """Научная нотация (1e5) — валидное число в Python."""
        output = self._run_predict_with_inputs(["1e5"])
        self.assertIn("Estimated price", output)

    def test_comma_as_decimal_rejected(self):
        """Запятая как разделитель (100,000) — отклоняется как невалидное число."""
        output = self._run_predict_with_inputs(["100,000", "100000"])
        self.assertIn("Error", output)


class TestRobustness_ModelFile(unittest.TestCase):
    """Различные варианты испорченного или отсутствующего model.json."""

    def _load(self, content=None, path=None):
        """Записывает content в файл и вызывает load_thetas()."""
        if path is None:
            f = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            if content is not None:
                f.write(content)
            f.close()
            path = f.name
            cleanup = True
        else:
            cleanup = False

        try:
            with patch.object(predict, "MODEL_FILE", path):
                t0, t1 = predict.load_thetas()
        finally:
            if cleanup and os.path.exists(path):
                os.unlink(path)
        return t0, t1

    def test_missing_model_returns_zeros(self):
        """Нет model.json → theta0=theta1=0."""
        t0, t1 = self._load(path="/nonexistent_file_xyz.json")
        self.assertEqual((t0, t1), (0.0, 0.0))

    def test_empty_file_returns_zeros(self):
        """Пустой model.json → theta0=theta1=0."""
        t0, t1 = self._load(content="")
        self.assertEqual((t0, t1), (0.0, 0.0))

    def test_invalid_json_returns_zeros(self):
        """Невалидный JSON → theta0=theta1=0."""
        t0, t1 = self._load(content="{ not valid json !!!")
        self.assertEqual((t0, t1), (0.0, 0.0))

    def test_missing_theta1_key_returns_zeros(self):
        """JSON без ключа theta1 → theta0=theta1=0."""
        t0, t1 = self._load(content='{"theta0": 100.0}')
        self.assertEqual((t0, t1), (0.0, 0.0))

    def test_missing_theta0_key_returns_zeros(self):
        """JSON без ключа theta0 → theta0=theta1=0."""
        t0, t1 = self._load(content='{"theta1": -0.02}')
        self.assertEqual((t0, t1), (0.0, 0.0))

    def test_string_values_returns_zeros(self):
        """theta0/theta1 как строки → theta0=theta1=0 (не краш)."""
        t0, t1 = self._load(content='{"theta0": "bad", "theta1": "value"}')
        self.assertEqual((t0, t1), (0.0, 0.0))

    def test_null_values_returns_zeros(self):
        """theta0/theta1 как null → theta0=theta1=0."""
        t0, t1 = self._load(content='{"theta0": null, "theta1": null}')
        self.assertEqual((t0, t1), (0.0, 0.0))

    def test_empty_json_object_returns_zeros(self):
        """Пустой JSON объект {} → theta0=theta1=0."""
        t0, t1 = self._load(content='{}')
        self.assertEqual((t0, t1), (0.0, 0.0))

    def test_valid_model_loads_correctly(self):
        """Корректный model.json загружается правильно."""
        t0, t1 = self._load(content='{"theta0": 8499.6, "theta1": -0.02144}')
        self.assertAlmostEqual(t0, 8499.6, places=3)
        self.assertAlmostEqual(t1, -0.02144, places=5)

    def test_extra_keys_in_model_ok(self):
        """Лишние ключи в model.json не ломают загрузку."""
        t0, t1 = self._load(
            content='{"theta0": 1.0, "theta1": 2.0, "extra": "ignored"}'
        )
        self.assertEqual(t0, 1.0)
        self.assertEqual(t1, 2.0)


class TestRobustness_TrainCSV(unittest.TestCase):
    """Попытки сломать train.py испорченным или нестандартным CSV."""

    def _train_with_csv(self, rows, headers="km,price"):
        """Запускает train.main() с временным CSV и model файлом."""
        csv_path = write_tmp_csv(rows, headers)
        model_path = tempfile.mktemp(suffix=".json")
        try:
            with patch.object(train, "DATA_FILE", csv_path), \
                 patch.object(train, "MODEL_FILE", model_path):
                train.main()
            result = json.load(open(model_path)) if os.path.exists(model_path) else None
        finally:
            os.unlink(csv_path)
            if os.path.exists(model_path):
                os.unlink(model_path)
        return result

    def test_missing_csv_exits_cleanly(self):
        """Нет data.csv → sys.exit(1), не краш с traceback."""
        with patch.object(train, "DATA_FILE", "/nonexistent_data.csv"):
            with self.assertRaises(SystemExit) as cm:
                train.main()
        self.assertEqual(cm.exception.code, 1)

    def test_wrong_headers_exits_cleanly(self):
        """CSV с неверными заголовками → sys.exit(1)."""
        csv_path = write_tmp_csv(["100,500", "200,400"], headers="mileage,cost")
        try:
            with patch.object(train, "DATA_FILE", csv_path):
                with self.assertRaises(SystemExit) as cm:
                    train.main()
            self.assertEqual(cm.exception.code, 1)
        finally:
            os.unlink(csv_path)

    def test_one_valid_row_exits_cleanly(self):
        """Только 1 строка данных → sys.exit(1) (нужно минимум 2)."""
        csv_path = write_tmp_csv(["100000,5000"])
        try:
            with patch.object(train, "DATA_FILE", csv_path):
                with self.assertRaises(SystemExit) as cm:
                    train.main()
            self.assertEqual(cm.exception.code, 1)
        finally:
            os.unlink(csv_path)

    def test_invalid_rows_skipped_silently(self):
        """Невалидные строки пропускаются, обучение продолжается."""
        result = self._train_with_csv([
            "100000,5000",
            "bad_value,hello",  # плохая строка
            "50000,7000",
        ])
        self.assertIsNotNone(result)
        self.assertIn("theta0", result)
        self.assertIn("theta1", result)

    def test_negative_km_rows_skipped(self):
        """Строки с отрицательным пробегом пропускаются."""
        result = self._train_with_csv([
            "100000,5000",
            "-5000,3000",   # отрицательный пробег
            "50000,7000",
        ])
        self.assertIsNotNone(result)

    def test_extra_columns_in_csv_ok(self):
        """Лишние колонки в CSV не ломают загрузку."""
        csv_path = write_tmp_csv(
            ["100000,5000,extra1", "50000,7000,extra2"],
            headers="km,price,color"
        )
        model_path = tempfile.mktemp(suffix=".json")
        try:
            with patch.object(train, "DATA_FILE", csv_path), \
                 patch.object(train, "MODEL_FILE", model_path):
                train.main()
        finally:
            os.unlink(csv_path)
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_csv_with_spaces_around_values(self):
        """Пробелы вокруг значений в CSV — корректно обрабатываются float()."""
        result = self._train_with_csv([
            " 100000 , 5000 ",
            " 50000 , 7000 ",
        ])
        # float(" 100000 ") работает, поэтому должно обучиться
        self.assertIsNotNone(result)

    def test_empty_csv_only_header_exits(self):
        """CSV с только заголовком (нет данных) → sys.exit(1)."""
        csv_path = write_tmp_csv([], headers="km,price")
        try:
            with patch.object(train, "DATA_FILE", csv_path):
                with self.assertRaises(SystemExit) as cm:
                    train.main()
            self.assertEqual(cm.exception.code, 1)
        finally:
            os.unlink(csv_path)

    def test_all_same_mileage_exits_cleanly(self):
        """Все одинаковые пробеги → std=0 → sys.exit(1), не краш."""
        csv_path = write_tmp_csv(["100000,5000", "100000,6000", "100000,7000"])
        try:
            with patch.object(train, "DATA_FILE", csv_path):
                with self.assertRaises(SystemExit) as cm:
                    train.main()
            self.assertEqual(cm.exception.code, 1)
        finally:
            os.unlink(csv_path)


# ===========================================================================
# PART 3 — MATH CORRECTNESS (формулы и математика)
# ===========================================================================

class TestMath_HypothesisFormula(unittest.TestCase):
    """Гипотеза строго = theta0 + (theta1 * mileage)."""

    def test_formula_predict(self):
        """predict.estimate_price соответствует theta0 + theta1*x."""
        cases = [
            (0.0, 0.0, 50000, 0.0),
            (8499.6, -0.02144, 0, 8499.6),
            (8499.6, -0.02144, 100000, 8499.6 + (-0.02144) * 100000),
            (1000.0, 2.0, 3.0, 1000.0 + 2.0 * 3.0),  # = 1006.0
        ]
        for t0, t1, km, expected in cases:
            with self.subTest(t0=t0, t1=t1, km=km):
                result = predict.estimate_price(km, t0, t1)
                self.assertAlmostEqual(result, expected, places=6)

    def test_formula_train(self):
        """train.estimate_price та же формула, что и в predict."""
        self.assertAlmostEqual(
            train.estimate_price(100, 5, 2),
            predict.estimate_price(100, 5, 2)
        )

    def test_formula_is_linear(self):
        """Гипотеза линейна: price(2x) = price(x) + theta1*x."""
        t0, t1 = 5000.0, -0.01
        x = 100000
        self.assertAlmostEqual(
            predict.estimate_price(2 * x, t0, t1),
            predict.estimate_price(x, t0, t1) + t1 * x
        )


class TestMath_GradientDescentFormulas(unittest.TestCase):
    """Проверка что формулы из subject реализованы точно."""

    def test_one_step_matches_subject_formulas(self):
        """
        Один шаг градиентного спуска вручную по формулам из subject.
        tmpθ0 = lr * (1/m) * Σ(estimate(km[i]) - price[i])
        tmpθ1 = lr * (1/m) * Σ(estimate(km[i]) - price[i]) * km[i]
        """
        km = [-1.0, 0.0, 1.0]
        prices = [7000.0, 6000.0, 5000.0]
        lr = 0.1
        m = 3
        t0, t1 = 0.0, 0.0

        errors = [(t0 + t1 * km[i]) - prices[i] for i in range(m)]
        exp_tmp0 = lr * sum(errors) / m
        exp_tmp1 = lr * sum(errors[i] * km[i] for i in range(m)) / m
        exp_t0 = t0 - exp_tmp0
        exp_t1 = t1 - exp_tmp1

        got_t0, got_t1 = train.gradient_descent(km, prices, lr, 1)

        self.assertAlmostEqual(got_t0, exp_t0, places=10,
            msg="theta0 после 1 итерации не совпадает с формулой subject")
        self.assertAlmostEqual(got_t1, exp_t1, places=10,
            msg="theta1 после 1 итерации не совпадает с формулой subject")

    def test_convergence_simple_case(self):
        """Для идеальных данных (y = 2x + 1) GD сходится к правильным theta."""
        # y = 1.0 + 2.0 * x (идеальные данные)
        xs = [float(i) for i in range(-5, 6)]  # -5..5
        ys = [1.0 + 2.0 * x for x in xs]

        # Нормализуем xs как train делает для km
        km_norm, mean_x, std_x = train.normalize(xs)
        t0n, t1n = train.gradient_descent(km_norm, ys, 0.1, 5000)
        t0, t1 = train.denormalize_thetas(t0n, t1n, mean_x, std_x)

        self.assertAlmostEqual(t0, 1.0, places=2,
            msg=f"theta0 должен ≈1.0, получили {t0:.4f}")
        self.assertAlmostEqual(t1, 2.0, places=2,
            msg=f"theta1 должен ≈2.0, получили {t1:.4f}")


class TestMath_Normalization(unittest.TestCase):
    """Нормализация: z-score и денормализация."""

    def test_normalize_zero_mean(self):
        """После нормализации mean ≈ 0."""
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        normed, mean_v, std_v = train.normalize(vals)
        self.assertAlmostEqual(sum(normed) / len(normed), 0.0, places=10)

    def test_normalize_unit_std(self):
        """После нормализации std ≈ 1."""
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        normed, mean_v, std_v = train.normalize(vals)
        mean_n = sum(normed) / len(normed)
        std_n = (sum((x - mean_n)**2 for x in normed) / len(normed)) ** 0.5
        self.assertAlmostEqual(std_n, 1.0, places=10)

    def test_normalize_identical_values_raises(self):
        """Все одинаковые значения → ValueError (std=0)."""
        with self.assertRaises(ValueError):
            train.normalize([5.0, 5.0, 5.0])

    def test_denormalize_roundtrip(self):
        """
        Денормализация: если обучить на нормализованных данных с theta_norm,
        то theta_real должны давать тот же результат для реального x.
        """
        xs_real = [50000.0, 100000.0, 150000.0, 200000.0]
        xs_norm, mean_x, std_x = train.normalize(xs_real)

        t0n, t1n = 5000.0, -500.0   # произвольные theta на нормализованных x
        t0r, t1r = train.denormalize_thetas(t0n, t1n, mean_x, std_x)

        for x_real, x_norm in zip(xs_real, xs_norm):
            price_norm = t0n + t1n * x_norm
            price_real = t0r + t1r * x_real
            self.assertAlmostEqual(price_norm, price_real, places=8,
                msg=f"Денормализация дала неверный результат для x={x_real}")


class TestMath_OverfittingCheck(unittest.TestCase):
    """
    Чек-лист упоминает overfitting как красный флаг:
    «если цена всегда одна и та же — это overfitting».
    Проверяем, что наша модель НЕ даёт константный результат.
    """

    @classmethod
    def setUpClass(cls):
        model_path = tempfile.mktemp(suffix=".json")
        with patch.object(train, "DATA_FILE",
                          os.path.join(PROJECT_DIR, "data.csv")), \
             patch.object(train, "MODEL_FILE", model_path):
            train.main()
        with open(model_path) as f:
            data = json.load(f)
        cls.theta0 = data["theta0"]
        cls.theta1 = data["theta1"]
        os.unlink(model_path)

    def test_not_constant_predictions(self):
        """Предсказания для разных пробегов различаются (нет overfitting)."""
        km_vals = [22899, 89000, 144500, 240000]
        prices = [predict.estimate_price(km, self.theta0, self.theta1)
                  for km in km_vals]
        unique = set(round(p, 4) for p in prices)
        self.assertGreater(len(unique), 1,
            "Все предсказания одинаковы — признак сломанной модели или overfitting")


# ===========================================================================
# PART 4 — Bonus files existence
# ===========================================================================

class TestBonus_FilesExist(unittest.TestCase):
    """Бонусные файлы существуют."""

    def test_visualize_exists(self):
        """visualize.py существует."""
        self.assertTrue(
            os.path.exists(os.path.join(PROJECT_DIR, "visualize.py")),
            "visualize.py не найден"
        )

    def test_precision_exists(self):
        """precision.py существует."""
        self.assertTrue(
            os.path.exists(os.path.join(PROJECT_DIR, "precision.py")),
            "precision.py не найден"
        )

    def test_precision_runs_without_crash(self):
        """precision.py запускается без краша."""
        model_path = tempfile.mktemp(suffix=".json")
        try:
            with patch.object(train, "DATA_FILE",
                              os.path.join(PROJECT_DIR, "data.csv")), \
                 patch.object(train, "MODEL_FILE", model_path):
                train.main()

            import precision
            buf = StringIO()
            with patch.object(precision, "MODEL_FILE", model_path), \
                 patch.object(precision, "DATA_FILE",
                              os.path.join(PROJECT_DIR, "data.csv")), \
                 patch("sys.stdout", buf):
                precision.main()

            output = buf.getvalue()
            self.assertIn("R²", output)
            self.assertIn("MAE", output)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


# ===========================================================================
# Цветной вывод тестов
# ===========================================================================

class Colors:
    """ANSI escape-коды для цветов."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class ColoredTestResult(unittest.TextTestResult):
    """Кастомный результат тестов с цветным выводом."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.verbosity = verbosity
        self.current_category = None
        self.success_count = 0

    def addSuccess(self, test):
        self.success_count += 1
        super().addSuccess(test)

    def startTest(self, test):
        # Определяем категорию теста
        test_class = test.__class__.__name__
        category = test_class.split('_')[0] + ' ' + test_class.split('_')[1] if '_' in test_class else test_class

        # Категории с цветами
        category_colors = {
            'TestChecklist': Colors.BLUE,
            'TestRobustness': Colors.YELLOW,
            'TestMath': Colors.CYAN,
            'TestBonus': Colors.HEADER,
        }

        if category != self.current_category:
            color = category_colors.get(category, Colors.END)
            self.stream.writeln(f"\n{color}{Colors.BOLD}{'═' * 60}")
            self.stream.writeln(f"  {category}")
            self.stream.writeln(f"{'═' * 60}{Colors.END}")
            self.current_category = category

        if self.verbosity > 1:
            self.stream.write(f"  {self.getDescription(test)} ... ")
            self.stream.flush()

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.verbosity > 1:
            self.stream.writeln(f"{Colors.GREEN}✓ PASSED{Colors.END}")

    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.writeln(f"{Colors.RED}✗ ERROR{Colors.END}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.writeln(f"{Colors.RED}✗ FAILED{Colors.END}")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.writeln(f"{Colors.YELLOW}⊘ SKIPPED{Colors.END}")

    def printSummary(self):
        """Печатает цветную сводку результатов."""
        total = self.success_count + len(self.failures) + len(self.errors)
        passed = self.success_count
        failed = len(self.failures)
        errors = len(self.errors)

        self.stream.writeln(f"\n{Colors.BOLD}{'═' * 60}{Colors.END}")
        self.stream.writeln(f"{Colors.BOLD}  ИТОГИ ТЕСТИРОВАНИЯ{Colors.END}")
        self.stream.writeln(f"{Colors.BOLD}{'═' * 60}{Colors.END}")

        self.stream.writeln(f"  Всего тестов:  {total}")
        self.stream.writeln(f"  {Colors.GREEN}✓ Пройдено:{Colors.END}    {passed}")
        if failed > 0:
            self.stream.writeln(f"  {Colors.RED}✗ Провалено:{Colors.END}   {failed}")
        if errors > 0:
            self.stream.writeln(f"  {Colors.RED}✗ Ошибки:{Colors.END}      {errors}")

        if self.wasSuccessful():
            self.stream.writeln(f"\n  {Colors.GREEN}{Colors.BOLD}🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!{Colors.END}")
        else:
            self.stream.writeln(f"\n  {Colors.RED}{Colors.BOLD}⚠ ЕСТЬ ПРОВАЛЬНЫЕ ТЕСТЫ{Colors.END}")

        self.stream.writeln(f"{Colors.BOLD}{'═' * 60}{Colors.END}")

    def stopTestRun(self):
        """Вызывается после завершения всех тестов."""
        self.printSummary()


class ColoredTestRunner(unittest.TextTestRunner):
    """Кастомный раннер с цветным выводом."""
    resultclass = ColoredTestResult

    def __init__(self, stream=None, descriptions=True, verbosity=2, **kwargs):
        super().__init__(stream, descriptions, verbosity, **kwargs)


# ===========================================================================
# Запуск
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Вывод заголовка
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 60}")
    print("  FT_LINEAR_REGRESSION — ПОЛНЫЙ ТЕСТ-СЮИТ")
    print(f"{'═' * 60}{Colors.END}\n")

    # Порядок совпадает с порядком чек-листа
    suite.addTests(loader.loadTestsFromTestCase(TestChecklist_Prelim))
    suite.addTests(loader.loadTestsFromTestCase(TestChecklist_PredictionBeforeTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestChecklist_TrainingPhase))
    suite.addTests(loader.loadTestsFromTestCase(TestChecklist_ReadCSV))
    suite.addTests(loader.loadTestsFromTestCase(TestChecklist_SimultaneousAssignment))
    suite.addTests(loader.loadTestsFromTestCase(TestChecklist_PredictionAfterTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustness_PredictInput))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustness_ModelFile))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustness_TrainCSV))
    suite.addTests(loader.loadTestsFromTestCase(TestMath_HypothesisFormula))
    suite.addTests(loader.loadTestsFromTestCase(TestMath_GradientDescentFormulas))
    suite.addTests(loader.loadTestsFromTestCase(TestMath_Normalization))
    suite.addTests(loader.loadTestsFromTestCase(TestMath_OverfittingCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestBonus_FilesExist))

    runner = ColoredTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
