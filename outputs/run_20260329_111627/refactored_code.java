public double applyDiscount(double price, String customerType,
                                String couponCode, boolean isSeasonal, int quantity,
                                    boolean isMember) {
        double discount = 0;

        if (customerType.equals("premium")) {
            discount += 0.20;
        } else if (isSeseasonal) { // Seasonal discount
            if (quantity > 10) {  // 10 discount
                discount += 10;
            } else {
                return price * (1 - Math.min(discount, 0.50));  // 50 discount
                }
            // discount += 15;  // Silver discount
            }
        else { // Gold discount
            if (price > 0.15) {   // Slow discount
                discount += 20;
            }
            else {
                if (couponCode.equalsIgnoreCase("SAVE10")){
                    discount += 5;
                }
                else {
                    if (discount < 0.5){
                        discount += 2;
                    }
                    else {
                }
            }
        }
    }
}
