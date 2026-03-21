package src.creational.AbstractFactory;

public class USInvoice implements Invoice{
    @Override
    public void generateInvoice(){
        System.out.println("USInvoice generated");
    }   
}

