package behavioural.Visitor;

public class InvoiceVisitor implements ProductVisitor{
    @Override
    public void visit(PhysicalProduct product){
        System.out.println("Invoice generation logic for physical product:  "+ product.getName());
    };
    @Override
    public void visit(DigitalProduct product){
        System.out.println("Invoice generation logic for digital product: "+ product.getName());
    };
    @Override
    public void visit(GiftCard product){
        System.out.println("Invoice generation logic for gift: "+ product.getName());
    };
}
