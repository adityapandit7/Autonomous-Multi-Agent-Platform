public class OrderService {

    public double calculateTotal(String itemType, int quantity,
                                 String promoCode, boolean isVip,
                                 String region, double taxRate) {

        double price = 0;

        if (itemType.equals("book")) {
            price = 12.99;
        } else if (itemType.equals("electronics")) {
            price = 299.99;
        } else if (itemType.equals("clothing")) {
            price = 49.99;
        } else if (itemType.equals("food")) {
            price = 5.99;
        } else {
            price = 9.99;
        }

        double subtotal = price * quantity;

        if (promoCode.equals("PROMO10")) {
            subtotal = subtotal * 0.90;
        } else if (promoCode.equals("PROMO20")) {
            subtotal = subtotal * 0.80;
        } else if (promoCode.equals("PROMO50")) {
            subtotal = subtotal * 0.50;
        }

        if (isVip) {
            subtotal = subtotal * 0.95;
        }

        double tax = subtotal * taxRate;

        if (region.equals("US")) {
            tax = tax * 1.02;
        } else if (region.equals("EU")) {
            tax = tax * 1.20;
        } else if (region.equals("UK")) {
            tax = tax * 1.15;
        }

        return subtotal + tax;
    }
}
