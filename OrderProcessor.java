public class OrderProcessor {

    /**
     * SMELLS: Long Method + Long Parameter List + Complex Conditional + Multifaceted Abstraction
     */
    public double calculateDiscount(double price, String customerType,
                                    int loyaltyYears, String membership,
                                    String couponCode, boolean isSeasonal,
                                    int bulkQuantity, boolean hasReferral) {
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

        if (loyaltyYears > 5) {
            discount += 0.05;
        } else if (loyaltyYears > 3) {
            discount += 0.03;
        } else if (loyaltyYears > 1) {
            discount += 0.01;
        }

        if (membership.equals("platinum")) {
            discount += 0.10;
        } else if (membership.equals("gold")) {
            discount += 0.05;
        }

        if (couponCode.equals("SAVE20")) {
            discount += 0.20;
        } else if (couponCode.equals("SAVE10")) {
            discount += 0.10;
        } else if (couponCode.equals("SAVE5")) {
            discount += 0.05;
        }

        if (isSeasonal) {
            discount += 0.15;
        }

        if (bulkQuantity > 10) {
            discount += 0.10;
        } else if (bulkQuantity > 5) {
            discount += 0.05;
        }

        if (hasReferral) {
            discount += 0.05;
        }

        double finalPrice = price * (1 - Math.min(discount, 0.50));
        System.out.println("Discount applied: " + (discount * 100) + "%");
        return finalPrice;
    }

    /**
     * SMELLS: Long Method + Multifaceted Abstraction
     */
    public void processOrder(String orderId, String customerId,
                             String[] items, double[] prices) {
        System.out.println("Processing order: " + orderId);

        double total = 0;
        for (int i = 0; i < items.length; i++) {
            total += prices[i];
        }

        System.out.println("Order total: " + total);

        if (total > 1000) {
            System.out.println("High value order - flagging for review");
        }

        System.out.println("Sending confirmation to customer: " + customerId);
        System.out.println("Updating inventory for order: " + orderId);
        System.out.println("Order " + orderId + " processed successfully");
    }
}
