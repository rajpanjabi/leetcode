Composite lets you treat a single object and a group of objects exactly the same way.

It is used when we have a hierarchical structure, when objects form a tree-like structure (e.g., folders inside folders, or products inside bundles).

When we want to treat individual and groups in the same way: When operations on single items and collections of items should be uniform (e.g., calculating total price, displaying structure).

An example of why we need it:

- We have a cart (Amazon), it can consist of either a Product or a Product Bundle. Now if I am dealing with the cart, I have to explicitly check if it's a product or bundle before doing anything on it. Also the logic of adding any object to cart has to be generalised to list<Object> so it can accept any of these types, which is not a good practice. When performing operations on these objects, I have to explicitly cast them to their specific type and then do getPrice or similar ops.

So, instead we define a common interface which is implemented by all the item types that go to our Cart. Now the client side doesn't have to care what type of object is added, because we know it will have to implement the logic defined in the interface.