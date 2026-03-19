package behavioural.Visitor;

public interface Product {
    void accept(ProductVisitor visitor);
}
