private double calculateDiscount(double price, String customerType,
                                 String couponCode, boolean isSeasonal,
                                 int quantity, boolean isMember) {
    double discount = 0;

    if (customerType.equals("premium")) {
        discount += 0.20;
    } else if (customerType.equals("gold")) {<beginofsentence>