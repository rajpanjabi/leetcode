package src.creational.AbstractFactory;
public class GSTInvoice implements Invoice{
    @Override
    public void generateInvoice(){
        System.out.println("GSTInvoice generated");
    }   
}

