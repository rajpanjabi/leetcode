package structural.Composite;

import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        // CartItem iphone = new Product(1500, "Iphone 17 Pro Max");
        CartItem charger = new Product(60, "Wireless Charger");
        CartItem earphones = new Product(250, "Airpods 3");
        
        CartItem book = new Product(50, "Atomic Habits");
        CartItem pencil = new Product(8, "Mechanical pencil");
        CartItem phonecase = new Product(30, "Bumper");
    
        ProductBundle bundle = new ProductBundle("Accessories");
        bundle.addItem(charger);
        bundle.addItem(earphones);
        bundle.addItem(phonecase);

        List<CartItem> items = new ArrayList<>();

        items.add(book);
        items.add(pencil);
        items.add(bundle);

        // Display cart
        System.out.println("Cart");
        double total=0;
        for (CartItem item :items){
            total+=item.getPrice();
            item.display();
        }
        System.out.println("Cart Total: " +total);
        

    }
    
}
