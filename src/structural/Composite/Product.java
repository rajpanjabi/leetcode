package structural.Composite;

public class Product implements CartItem{
    private double price;
    private String description;
    public Product(double price, String description){
        this.price=price;
        this.description=description;
    }
    @Override
    public double getPrice(){
        return price;
    }

    @Override
    public void display(){
        System.out.println("Product: "+description+" Price: "+ price);
    }
}
