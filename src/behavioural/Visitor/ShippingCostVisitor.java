package behavioural.Visitor;

public class ShippingCostVisitor implements ProductVisitor{
    @Override
    public void visit(PhysicalProduct product){
        System.out.println("Shipping cost logic for physical product: "+ product.getName());
    };
    @Override
    public void visit(DigitalProduct product){
        System.out.println("Shipping cost logic for digital product: "+ product.getName());
    };
    @Override
    public void visit(GiftCard product){
        System.out.println("Shipping cost logic for gift: "+ product.getName());
    };
}

    

