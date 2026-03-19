package behavioural.Visitor;

public class PhysicalProduct implements Product{

    private String name;
    private double price;

    public PhysicalProduct(String name, double price){
        this.name=name;
        this.price=price;
    }
    public String getName(){
        return this.name;
    }
    @Override
    public void accept(ProductVisitor visitor){
        visitor.visit(this);
    }
}
