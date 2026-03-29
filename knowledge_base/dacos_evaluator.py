# prompt_engine/dacos_evaluator.py
#
# DACOS Evaluator — benchmarks the system against the DACOS dataset.
# Used to measure smell detection accuracy using real annotated Java samples.

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DACOSEvaluator:
    """
    Evaluates the smell detector against the DACOS dataset.
    If the dataset is not available, uses built-in Java sample data.
    """

    def __init__(self, dacos_path: Optional[str] = None):
        self.dacos_path = Path(dacos_path) if dacos_path else None
        self.dataset = []
        self.results = []

        if self.dacos_path and self.dacos_path.exists():
            self._load_dataset()

    def _load_dataset(self):
        if self.dacos_path.suffix == ".json":
            self._load_json()
        elif self.dacos_path.suffix == ".csv":
            self._load_csv()
        elif self.dacos_path.suffix == ".sql":
            logger.info("SQL file detected — export to CSV/JSON first for evaluation use.")
            self._load_sql_instructions()
        else:
            self._scan_directory()

    def _load_json(self):
        try:
            with open(self.dacos_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.dataset = data if isinstance(data, list) else data.get("samples", [])
            logger.info(f"Loaded {len(self.dataset)} samples from JSON")
        except Exception as e:
            logger.error(f"Failed to load JSON: {e}")

    def _load_csv(self):
        try:
            with open(self.dacos_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.dataset = list(reader)
            logger.info(f"Loaded {len(self.dataset)} samples from CSV")
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")

    def _scan_directory(self):
        for ext, loader in [("*.json", self._load_json), ("*.csv", self._load_csv)]:
            for file in self.dacos_path.glob(ext):
                self.dacos_path = file
                loader()
                return

    def _load_sql_instructions(self):
        logger.info("To use DACOS SQL: import into SQLite, export tables to CSV, point evaluator to CSV.")

    def create_test_samples(self, count: int = 10) -> List[Dict]:
        if not self.dataset:
            return self._create_sample_data(count)
        samples = []
        for i, item in enumerate(self.dataset[:count]):
            code  = self._extract_code(item)
            label = self._extract_label(item)
            if code and label:
                samples.append({"id": i, "code": code, "expected_smell": label, "original_data": item})
        return samples

    def _extract_code(self, item: Dict) -> Optional[str]:
        for field in ["code", "source", "method_code", "content", "text"]:
            if field in item and item[field]:
                return item[field]
        return None

    def _extract_label(self, item: Dict) -> Optional[str]:
        for field in ["smell", "label", "smell_type", "type", "category"]:
            if field in item and item[field]:
                return item[field]
        return None

    def _create_sample_data(self, count: int) -> List[Dict]:
        """
        Built-in Java sample data for when no DACOS dataset is available.
        All samples are valid Java — NOT Python.
        """
        logger.warning("No DACOS dataset found. Using built-in Java sample data.")

        samples = [
            {
                "code": """public double processOrder(String orderId, String customerId,
                        String productCode, double price, int quantity,
                        String couponCode, boolean isPriority, String shippingZone,
                        String paymentMethod, boolean giftWrap) {
    double total = price * quantity;
    if (couponCode.equals("SAVE10")) total *= 0.9;
    return total;
}""",
                "expected_smell": "Long Parameter List"
            },
            {
                "code": """public void handleCustomerRequest(String customerId) {
    // Validate customer
    if (customerId == null || customerId.isEmpty()) {
        System.out.println("Invalid customer");
        return;
    }
    // Fetch data
    String data = fetchFromDatabase(customerId);
    // Process data
    double result = 0;
    for (String item : data.split(",")) {
        result += Double.parseDouble(item.trim());
    }
    // Apply business logic
    if (result > 1000) {
        result *= 0.9;
    } else if (result > 500) {
        result *= 0.95;
    } else if (result > 200) {
        result *= 0.98;
    }
    // Log and notify
    System.out.println("Result: " + result);
    sendEmailNotification(customerId, result);
    updateDatabase(customerId, result);
    generateReport(customerId, result);
}""",
                "expected_smell": "Long Method"
            },
            {
                "code": """public double calculateDiscount(String customerType, String membershipLevel,
                                             String couponCode, boolean isSeasonal) {
    double discount = 0;
    if (customerType.equals("premium")) {
        discount += 0.20;
    } else if (customerType.equals("gold")) {
        discount += 0.15;
    } else if (customerType.equals("silver")) {
        discount += 0.10;
    } else if (customerType.equals("bronze")) {
        discount += 0.05;
    }
    if (membershipLevel.equals("platinum")) {
        discount += 0.10;
    } else if (membershipLevel.equals("gold")) {
        discount += 0.05;
    }
    if (couponCode.equals("SAVE20")) {
        discount += 0.20;
    } else if (couponCode.equals("SAVE10")) {
        discount += 0.10;
    }
    if (isSeasonal) {
        discount += 0.15;
    }
    return Math.min(discount, 0.50);
}""",
                "expected_smell": "Complex Conditional"
            },
            {
                "code": """public void processAndSaveOrder(String orderId, String[] items, double[] prices) {
    // Calculate total (responsibility 1: calculation)
    double total = 0;
    for (int i = 0; i < prices.length; i++) total += prices[i];
    // Log the order (responsibility 2: I/O)
    System.out.println("Order " + orderId + " total: " + total);
    // Send notification (responsibility 3: communication)
    sendConfirmationEmail(orderId);
    // Save to database (responsibility 4: persistence)
    saveToDatabase(orderId, total);
}""",
                "expected_smell": "Multifaceted Abstraction"
            },
        ]
        return samples[:min(count, len(samples))]

    def evaluate_smell_detection(self, smell_detector, parsed_code_func) -> Dict:
        """Evaluate smell detection accuracy against test samples."""
        test_samples = self.create_test_samples(20)
        results = {
            "total": len(test_samples), "correct": 0, "incorrect": 0,
            "by_smell": {}, "details": []
        }

        for sample in test_samples:
            try:
                parsed  = parsed_code_func(sample["code"])
                detected = smell_detector.detect_smells(parsed)
                top_smell = detected[0]["name"] if detected else "None"
                expected  = sample["expected_smell"]
                is_correct = (top_smell == expected)

                if is_correct: results["correct"] += 1
                else:          results["incorrect"] += 1

                results["by_smell"].setdefault(expected, {"total": 0, "correct": 0})
                results["by_smell"][expected]["total"] += 1
                if is_correct:
                    results["by_smell"][expected]["correct"] += 1

                results["details"].append({
                    "sample_id":    sample.get("id", 0),
                    "expected":     expected,
                    "detected":     top_smell,
                    "correct":      is_correct,
                    "all_detected": [s["name"] for s in detected]
                })
            except Exception as e:
                logger.error(f"Error evaluating sample: {e}")

        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]
            for smell, data in results["by_smell"].items():
                if data["total"] > 0:
                    data["accuracy"] = data["correct"] / data["total"]

        return results

    def save_evaluation_report(self, results: Dict, output_path: Optional[str] = None):
        if output_path is None:
            output_path = f"dacos_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            summary_path = output_path.replace(".json", "_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(self._format_summary(results))
            logger.info(f"Evaluation report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation: {e}")

    def _format_summary(self, results: Dict) -> str:
        lines = ["=" * 60, "DACOS EVALUATION SUMMARY", "=" * 60, ""]
        lines.append(f"Total samples    : {results.get('total', 0)}")
        lines.append(f"Correct          : {results.get('correct', 0)}")
        lines.append(f"Incorrect        : {results.get('incorrect', 0)}")
        if "accuracy" in results:
            lines.append(f"Overall accuracy : {results['accuracy'] * 100:.2f}%")
        lines += ["", "-" * 40, "Per-Smell Accuracy:", "-" * 40]
        for smell, data in results.get("by_smell", {}).items():
            if data["total"] > 0:
                acc = data.get("accuracy", data["correct"] / data["total"]) * 100
                lines.append(f"  {smell}: {data['correct']}/{data['total']} ({acc:.2f}%)")
        lines += ["", "=" * 60]
        return "\n".join(lines)
