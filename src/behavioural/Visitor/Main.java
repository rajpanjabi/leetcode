package behavioural.Visitor;

import java.util.ArrayList;

public class Main {

    public static void main (String [] args){
        // First lets create a list of products
        ArrayList<Product> products = new ArrayList<>();
        Product physical = new PhysicalProduct("Iphone", 1599.99);
        Product digital = new DigitalProduct("Game Of Thrones eBook", 69.50);
        Product giftCard = new GiftCard("Mastercard", 199.99);
        products.add(physical);
        products.add(digital);
        products.add(giftCard);

        // Now we want to generate invoice and also calculate shipping cost for each of the product, but we
        // dont have the logic in their respective class, instead in the visitor class

        // We have to create instance of these visitors 
        ProductVisitor invoiceVisitor= new InvoiceVisitor();
        ProductVisitor shippingCostVisitor= new ShippingCostVisitor();
        for (Product p : products){
            p.accept(invoiceVisitor);
            p.accept(shippingCostVisitor);
            System.out.println("");
        }
        
        
    }
    
}
