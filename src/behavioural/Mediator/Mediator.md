
Mediator pattern is used when we have multiple objects and they want to communicate with each other. For instance, let's say we have multiple plane objects, if one plan wish to lands, it has to pass this message to all the other planes planning to land to avoid crash, if there was no medium between them, then these plane objects would be tightly coupled with each other, adding new objects or removing would be challenging.

So we create a Mediator that handles the communication.