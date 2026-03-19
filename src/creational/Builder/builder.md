**Builder Pattern**

    Builder Pattern is used to simplify the object creation process for objects with multiple parameters.
    
    It is very difficult to remember the order and also only some of the parameters are actually requried, most are optional, but to create an object you need to explicitly pass all the params (null or values).

To solve this one of the tricks could be to do constructor overloading, that is create multiple cosntructors with different combinations of params like Burger(type, bun), Burger(type, bun, hasSide),  Burger(type, bun, hasSide, hasDrink),  Burger(type, bun, hasSide, hasDrink, hasCheese) and so on.. this is not readable and not ideal in case of more than 3-5 parameters.

```java
// Telescoping Constructor Anti-Pattern
// To manage optional parameters, many developers try to solve this by writing multiple overloaded constructors — each with one more optional parameter than the last. For example:
// Java

class BurgerMeal {
    public BurgerMeal(String bun, String patty) { ... }
    public BurgerMeal(String bun, String patty, boolean cheese) { ... }
    public BurgerMeal(String bun, String patty, boolean cheese, String side) { ... }
    public BurgerMeal(String bun, String patty, boolean cheese, String side, String drink) { ... }
}
```

So here we use Builder pattern

For the Main object we make the constructor private and it only takes a builder object. We create a static nested inner Builder class which creates a builder object with the required params. We make multiple 

Lombok is a Java library that reduces boilerplate code using annotations. One of its popular features is the @Builder annotation, which automatically generates a builder class behind the scenes.

Instead of writing the builder logic manually, you just annotate your class:
```Java

@Builder
public class User {
    private String name;
    private int age;
    private String address;
}

// Now, you can build objects using a fluent API:
User user = User.builder()
            .name("John")
            .age(30)
            .address("NYC")
            .build();
            
```


```java
class BurgerMeal{

    // some required params and some optional
    // the attributes of burger object would be private and final, so once configured cannot be modified
    private final String burgerType;
    private final String bunType;
    private final boolean hasSide;
    private final boolean hasDrink;
    private final String pattyType;
    private BurgerMeal(BurgerBuilder builder){
        this.burgerType=builder.burgerType;

    }
    static class BurgerBuilder{
        private final String burgerType;
        private final String bunType;
        private final boolean hasSide;
        private final boolean hasDrink;
        private final String pattyType;

        public BurgerBuilder(String burgerType, String pattyType){
            this.burgerType=burgerType;
            this.pattyType=pattyType

        }


    }
}
```