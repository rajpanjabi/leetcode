package behavioural.Visitor;

public class GiftCard implements Product{

    private String name;
    private double price;

    public GiftCard(String name, double price){
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
