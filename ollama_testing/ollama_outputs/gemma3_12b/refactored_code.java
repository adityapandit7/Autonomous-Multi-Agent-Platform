public double applyDiscount(double price, String customerType,
                                String couponCode, boolean isSeasonal,
                                int quantity, boolean isMember) {
        double discount = calculateCustomerDiscount(customerType);
        discount += calculateCouponDiscount(couponCode);
        discount += (isSeasonal  0.15 : 0.0);
        discount += (quantity > 10  0.10 : 0.0);
        discount += (isMember  0.05 : 0.0);

        return price * (1 - Math.min(discount, 0.50));
    }

private double calculateCustomerDiscount(String customerType) {
        double discount = 0.0;
        if (customerType.equals("premium")) {
            discount += 0.20;
        } else if (customerType.equals("gold")) {
            discount += 0.15;
        } else if (customerType.equals("silver")) {
            discount += 0.10;
        }
        return discount;
    }

private double calculateCouponDiscount(String couponCode) {
        double discount = 0.0;
        if (couponCode.equals("SAVE10")) {
            discount += 0.10;
        } else if (couponCode.equals("SAVE5")) {
            discount += 0.05;
        }
        return discount;
    }