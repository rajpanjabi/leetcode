package behavioural.Visitor;

public interface ProductVisitor {
    void visit(PhysicalProduct product);
    void visit(DigitalProduct product);
    void visit(GiftCard product);
}
