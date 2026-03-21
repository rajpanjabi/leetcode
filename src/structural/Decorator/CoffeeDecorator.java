package structural.Decorator;

// this is the base decorator which is further implemented by decorator classes
public abstract class CoffeeDecorator implements Coffee{
    // We make this class abstract and delegate the implementation to the child classes.
    Coffee coffee;
    // We use a base coffee object
    public CoffeeDecorator(Coffee coffee){
        this.coffee=coffee;
    }
}
