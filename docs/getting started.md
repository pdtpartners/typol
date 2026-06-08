---
weight: -1
---

# Getting Started

See the [*How it works* in the README](../#how-it-works) for a quick introductory walkthrough. Typol should *mostly* work like Polars, so if you're stuck on how to write something, following docs for Polars should be a great place to start!

## Step-by-step guide

### Defining your shapes

Before you can operate on your data, you'll need to declare what it looks like. In Typol, we do this using a `Shape`:

```py
import typol as tp

class Purchase(tp.Shape):
    customer = tp.dimension(str)
    product = tp.dimension(str)
    price = tp.dimension(float)
    quantity = tp.dimension(int)
    date = tp.dimension(datetime.date)
```

We can use int or string enums to define fixed values:

```py
class PaymentMethod(Enum):
    CARD = "card"
    CASH = "cash"


class Purchase(tp.Shape):
    ...  # same as above
    method = tp.dimension(PaymentMethod)
```

We should define shapes for all the fixed data we have, and we can combine them together to form new shapes:

```py
class Customer(tp.Shape):
    name = tp.dimension(str)
    age = tp.dimension(int)
    phone = tp.dimension(int)
    state = tp.dimension(str)

# You can use `&` to quickly combine shapes
CustomerPurchase = Customer & Purchase

# Or you can use multiple inheritence and add more fields at the same time
class FullPurchaseInfo(Customer, Purchase):
    gift_card_used = tp.dimension(bool)
```

### Reading in your data

Now we've got some shapes, we can pull in some data. There are multiple ways to construct a Typol frame depending on where your data currently is. If you have the choice, use `tp.Entry` or `.set_all` since these provide typing guarantees!

- **From Polars**—Let's say you already have a Polars dataframe. It's easy to convert it into a Typol frame by just passing it into the constructor:

```py
purchases = tp.DataFrame(Purchase, pl_df)
```

- **From dictionaries**—If you already have a list of dictionaries, you can do the same as with Polars

```py
purchases = tp.DataFrame(Purchase, list_of_dicts)
```

- **Using `tp.Entry`**—A more typed solution is to use `tp.Entry`, which enforces that all the fields set a valid for the frame, and that the fields are set to values of the correct type

```py
purchases = tp.DataFrame(
    customers,
    (
        tp.Entry.of(
            Customer.name.set(r.first_name + " " + r.last_name),
            Customer.age.set(r.age),
            Customer.state.set(r.state_code),
        )
        for r in db_results
    )
)
```

- **From columnar data**—If the data is in columns rather than rows, you can construct the frame by using `.set_all` for each column:

```py
people = tp.DataFrame(
    Person,
    (
        Person.age.set_all([25, 22, 35, 24, 35, 36]),
        Person.name.set_all(["Alice", "Bob", "Charles", "David", "Eve", "Fred"]),
    ),
)
```

- **Reading from a file**—To read from a file, use `tp.read_csv`. This will handle any differences in casing or spacing for you:

```py
# customers.csv:
#   Name,Age,Phone,State
#   Alice,21,12345678,NY
#   Bob,22,87654321,CA
customers = tp.read_csv(Customers, "customers.csv")
```


### Transforming your data

#### Updating values

If we're just trying to update values and not add or remove columns, you can use `df.with_columns`:

```py
# You can use customers.s.<col> and Customer.<col> pretty interchangeably
customers = customers.with_columns(customers.s.age + 1)  # Everyone gets a year older

# Move the quantity into the price and by one of "Product x2" rather than two "Product"s
purchase = purchase.with_columns(
    # Note, the _out variant of a method will lose its shape, compare `cast` that can go to other
    # representations of the same type (i.e. INT32 to INT64), with `cast_out` that can change types
    purchase.s.name + " x" + purchase.s.quantity.cast_out(int),
    purchase.s.quantity.set(1),  # This is a shortcut for `tp.lit(1).to(purchase.s.quantity)`
    purchase.s.price * purchase.s.quantity,
)
```

If your expression changes types, you will no longer be able to assign it to the same column, so you'll have to give it a new column assignment using `.to(...)`:

```py
# Name everyone with their phone number
customers.with_columns(customers.s.phone.cast_out(str).to(customers.s.name))

customers.with_columns(
    # Even though this is string to string, it uses a list in the middle so loses its column, so we
    # need to map it back with `.to(customer.s.name)`
    customers.s.name.str.split(" ").list.join("_").to(customer.s.name),
)
```

If you forget to, you'll see a type error like this:

```py
# (1) Expected `Expr[Customer, Customer, Any]`, found `Expr[Customer, Never, str]`
customers.with_columns(customers.s.phone.cast_out(str))
```

#### Filtering values

Filtering is pretty simple, you just need to create a boolean expression:

```py
customers.filter(customers.s.age >= 20, customers.s.state != "NY" | customers.s.name.str.starts_with("J"))
```

#### Adding or removing columns

You'll need to define a shape to have any extra columns, and if you want to properly remove a column (rather than `null`ing it with `.with_columns(col.null())`) you'll also need to define a shape. The easiest way is to extend a current shape:

```py
class PurchaseAtStore(Purchase):
    location: tp.dimension(str)

purchases = purchases.transform(
    PurchaseAtStore, 
    tp.when(purchases.s.product.str.contains("Ice Cream"))
    .then("Beach")
    .otherwise("Downtown")
    .to(PurchaseAtStore.location)
)
```

Most commonly, you'll need to add or remove columns when joining data. This will happen automatically for you, constructing `LeftShape & RightShape`:

```py
customer_purchases = purchases.join(
    customers,
    purchases.s.customer.on(customers.s.name),
    how="left",
)
```

If there are conflicts when joining columns, you can add suffixes to one of the frames to refer to them independently:

```py
# Add a suffix to all the columns so they can be referred to independently
other_customers = customers.suffix()
# Join customers against itself to find ones where the names conflict
customers_with_the_same_name = customers.join(
    other_customers,
    # To refer to suffixed columns, do suffixed_shape(original_column)
    customers.s.name.on(other_customers.s(Customer.name))
    how="cross"
).filter(customer.s.phone != other_customers.s(Customer.phone))
```


#### Aggregating values


```py
# By default, this will group by all unspecified columns
total_purchases = purchases.agg(
    purchases.c.quantity.sum(),
    # Calculate a weighted average by multiplying by the quantity, and dividing at the
    # end by the summed quantity
    purchases.c.price * purchases.c.quantity,
).with_columns(purchases.c.price / purchases.c.quantity)

# If you just want to group by a particular column, you can specify it explicitly
spend_per_customer = purchases.group_by(Purchase.customer).agg(
    (Purchase.price * Purchase.quantity).sum(),
    Purchase.quantity.sum(),
)
```

### Getting out your data

- To go back to Polars, just use `my_typol_df.dataframe`. This can be especially helpful for intermediate operations where typing might not make sense
- To get each row, use `df.iter_rows()`:

```py
insert_into_db(
    CustomerObj(
        name=row[Customer.name],  # Each of these is correctly typed as `str` or `int`, so is type checked
        age=row[Customer.age],
        state=row[Customer.state],
    )
    for row in customers.iter_rows()
)
```

- To get a column, you can access a series and convert it into a list:

```py
customer_names = purchases[purchases.s.customer].unique().to_list()  # type inferred as list[str]
```

- Often you want to write it out to a CSV, you can simply do this with `.write_csv`, but if you want more control you can use `write_csv_of`:

```py
purchases.write_csv()

purchases.write_csv_of(
    # `to_out` lets us leave the shape and create arbitrary names
    purchases.c.customer.to_out("Customer Name"),
    purchases.c.product.to_out("Product Name"),
)
```

## Concepts that are different to Polars

Apart from the obvious `tp.Shape` differences, here are a couple things you should be aware of coming from Polars

### Expression targets

Polars expressions have names, which you can set with `.alias`. In Typol, these targets are associated with shapes, i.e.

```py
# This expression is read from a `Customer`, targets a `Customer`, and is a `str`
Customer.name  # Expr[Customer, Customer, str]
# This expression is read from a `Customer`, targets a `Purchase`, and is an int`
(tp.lit(2026) - Purchase.age).to(Customer.year_of_birth)  # Expr[Customer, Purchase, int]
```

When expressions change type, they are no longer valid for their original shape:

```py
# This expression is read from a `Customer`, but there is no longer valid for a shape, and is a `bool`
Purchase.age > 21  # Expr[Purchase, Never, bool]
```

This is fine in some situations, like filtering, where the output is not written to a shape. Note, only the leftmost expression is used for a name, so if an expression is combined with others which are valid for a shape, then it doesn't matter that it itself has lost its shape:

```py
# Filtering doesn't write to a column, so the output doesn't need to be valid for the original shape
df.filter(Purchase.age > 21)
# The left expression already gives it a valid `bool` column name (`is_legal`), so the overall expression is ok to write to `Purchase`
df.with_columns(Purchase.is_legal & (Purchase.age > 21))
# This is ok, since Purchase.name is valid for a `str`, so it doesn't matter than the casted phone number isn't valid for its shape
df.with_columns(Purchase.name.fill_null(Purchase.phone_number.cast_out(str)))
```

If an expression has lost its shape, you can use `to` to reassociate it with a valid column

```py
# (Purchase.age > 21) is a boolean so is not valid for `age`, but it's total appropriate for `is_legal`
df.with_columns((Purchase.age > 21).to(Purchase.is_legal))
# The left expression here has lost its shape thanks to being casted from an int to a str, so we reassociate it with a valid str column
df.with_columns(Purchase.phone_number.cast_out(str).fill_null(Purchase.name).to(Purchase.name))
```

#### `to`

Relabelling columns is quite important in Typol, since shapes are fixed and intermediate expressions need to be repointed to a column in the resultant shape. For this reason, `.alias` is simply called `.to` in Typol, to be concise and make it clear we're not creating another name for it, rather matching it to which column it should end up in.

```py
# Polars
(pl.col.age + 1).alias("age_next_year")

# Typol
(person.s.age + 1).to(FuturePerson.age_next_year)
```

#### `*_out`

Some methods end with `_out`, i.e. `map_out` and `replace_out`. These are variants of the core methods that lose their shape:

```
# Doesn't need to lose its shape since it stays a string in the name column
Person.name.replace(nicknames)  # Expr[Person, Person, str]
# Loses its shape (i.e. goes to `Never`) since if `phone_number = tp.dimension(int)`, it cannot contain a string
Person.phone_number.replace_out(region_name)  # Expr[Person, Never, str]
```

You can use [`.to`][typol.expr.Expr.to] to bind these expressions to valid columns in a new shape

```py
Person.phone_number.replace_out(region_name).to(Region.name)  # Expr[Person, Region, str]
```

### `transform`

In Polars we use select and with_columns, in Typol we need to be explicit about which shape we're going to, so `transform` fills the role of these. `with_columns` is still helpful when no new columns are being added.

```py
# Polars
customers.select(pl.col.name, pl.col.age)
customers.select(pl.col.str.strip_chars(), pl.col.age + 1)
customers.with_columns(pl.col.str.strip_chars(), pl.col.age + 1)

# Typol
class Person(tp.Shape):
    name = tp.dimension(str)
    age = tp.dimension(int)

# Typol will always select all available columns
customers.transform(Person)
customers.transform(Person, customers.s.name.str.strip_chars(), customers.s.age + 1)
customers.with_columns(customers.s.name.str.strip_chars(), customers.s.age + 1)
```

#### `*_transform` helpers

Some operations are more ergonomic when you can switch between shapes and transform values at the same time. For example, an aggregation is likely to need different columns than the original data, potentially dropping columns or creating list columns:

```py
class AgesOfNames(tp.Shape):
    name = tp.dimension(str)
    ages = tp.dimension(tp.list_of(str))


name_stats = people.group_by_transform(
    AgesOfNames, 
    people.s.name.to(AgesOfNames.name)
).agg(people.s.ages.agg().to(AgesOfNames.ages))
```

This avoids having to have awkward states, i.e. using strings to represent lists or singleton lists before aggregation, and can avoid having more intermediate shapes