# Typol

A typed wrapper around Polars, for statically enforcing shape types for dataframes. Get the speed and algebra of dataframes with the guarantees and maintainability of static typing. Follow how it works below, or scoll down to [see a full example](#full-example)

* Built around [Polars](https://github.com/pola-rs/polars/) – a thin layer to keep it simple and unsurprising; Polars docs will mostly apply outside of core type concepts
* Statically typed dataframes:
  * Build confidence before running code: points out errors before they happen
  * Allows tooling to guide you: language servers know what columns are available and what their types are, so what operations can happen
  * Keeping structure also makes it easier to dive into long-untouched code, and can enforce consistency across systems
  * *[See full reasoning in our FAQs](#faqs)*
* [Ty](https://github.com/astral-sh/ty) type checking — built to take advantage of the latest features like intersection types


### How it works

Define your `Shape`s with the same goal as when you'd define a `Schema` in Polars. Let's say you have a dataframe that has 4 columns:

```python
class Account(tp.Shape):
    name = tp.dimension(str)
    website = tp.dimension(str)
    account_age = tp.dimension(datetime.timedelta)
    uid = tp.dimension(int)

accounts = tp.DataFrame(Account, [...])  # type: tp.DataFrame[Account]
```

You can write well-typed expressions to operate on this data:

```python
email_address = accounts.s.name.str.to_lowercase() + "@" + accounts.s.website  # type: Expr[Account, Account, str]
```

The above says it's an expression that requires an `Account` dataframe, can write to an `Account` dataframe, and results in a `str` value. If you make a type error it'll tell you:

```python
# Unsupported `+` operation ty(unsupported-operator)
# example.py(_, 17): Has type `BoundDimension[Account, int]`
# example.py(_, 42): Has type `Literal["@"]`
email_address = accounts.s.uid + "@" + accounts.s.website
```

because you're trying to add a string and an int!

You can create new dataframes with these transformations:

```python
class Contact(tp.Shape):
    uid = tp.dimension(int)
    email = tp.dimension(str)

accounts.transform(Contact, email_address.to(Contact.email))
```

but it will catch you if you don't put it an appropriate column for the new shape:

```python
# (1) Expected `Expr[Account, Contact, Any]`, found `Expr[Account, Account, str]`
accounts.transform(Contact, email_address)
# (2) Argument to bound method `Expr.to` is incorrect: Expected `BoundDimension[Contact, str]`, found `BoundDimension[Contact, int]`
accounts.transform(Contact, email_address.to(Contact.uid))
```

since for (1) email address is still based off `Account.name`, and we haven't assigned it to something valid for `Contact`, and for (2) we can't assign a `str` to `Contact.uid`, which has to be an int. 

You can also update columns in the same shape:

```python
accounts.with_columns(accounts.s.name.str.to_lowercase())
```

which will have lowercased names, or you can add a column to a shape:

```python
accounts_with_emails = accounts.with_columns(
    (accounts.s.name.str.to_lowercase() + "@" + accounts.s.website).alias("email")
)
# It tracks your added column name and reveals the correct type:
reveal_type(accounts_with_emails.s.email)  # Revealed type is `BoundDimension[..., str]`
```


If you change type, it won't be assignable to the same column:

```python
# `.uid.cast_out(str)` can't be left in `.uid` if cast to a str
# Argument to bound method `DataFrame.with_columns` is incorrect: 
#  Expected `Expr[Account, Account, _] | BoundSeries[Account, _]`, found `Expr[Account, Never, str]`
accounts.with_columns(accounts.s.uid.cast_out(str))
```

so you have to assign it using `.to` to one that makes sense:

```python
# This works fine, because `str`s make sense for `.name` 
accounts.with_columns(accounts.s.uid.cast_out(str).to(accounts.s.name))
```
Need to filter out only interesting rows? No problem:

```python
# Note: `accounts.s.website == "interesting.com"` has type `Expr[Account, _, bool]`
accounts.filter(accounts.s.website == "interesting.com")
```

and it catches it if your filter makes no sense:

```python
# Argument to bound method `DataFrame.filter` is incorrect: Expected `Expr[Account, _, bool]`, found `Expr[Account, _, str]`
accounts.filter(accounts.s.website + "interesting.com")
# Argument to bound method `DataFrame.filter` is incorrect: Expected `Expr[Account, _, bool]`, found `Expr[Contact, _, bool]`
accounts.filter(contacts.s.email.str.ends_with("interesting.com"))
```

The first isn't a boolean expression to filter on (just a `str`), and the second tries to filter on a column of a different shape we don't have.

It even tracks combined shapes when you join data together!

```python
full_data = accounts.join(contacts, accounts.s.uid.on(contacts.s.uid))  # type: tp.DataFrame[Accounts & Contacts]
```

And when you get extract data back into Python types, it knows what it should be:

```python
full_data[full_data.s.email].to_list()  # type: list[str]
full_data[full_data.s.uid].to_list()  # type: list[int]

combined_ages = datetime.timedelta()
for row in full_data.iter_rows():
    combined_ages += row[full_data.s.account_age]  # type: datetime.timedelta
```

[See full example](#full-example)

As much of the above is statically enforced as is possible, giving much greater guarantees for
dataframe code. Where static enforcement is not possible, dynamic enforcement is used to ensure
the static types are always correct

## FAQs

#### Why do you need static checking of dataframes?

Dataframe code can be difficult to maintain, as the shape is often quite implicit. Columns can be added and removed ad-hoc, with different sections of code having different expectations, and no way to enforce consistency statically

#### Why not use Pandera/other dynamic dataframe checking?

Existing tools, like Pandera, do provide dynamic verification of dataframe shapes. Whilst this can be good, it bites you at runtime which is well after a problem should be caught. It also doesn't provide any tooling benefit: types guide development by enabling autocomplete and the language server to direct you to what should and shouldn't work interactively. This hasn't been the direction the dataframes community has headed since it's data science focused, where the shapes are much more ad-hoc and transient, but fits well for code that is already well structured for dataclasses, like application level code

Simply, without static checking:
  * there's no enforcement around dataframe expressions, only the shapes
  * there's no enforcement going between dataframe and non-dataframe code
  * you have to write enough tests to cover all the cases, since the shape enforcement requires the code to be run (such as in a test)
  * your tooling doesn't help guide you as you develop

#### Why not just write code that uses dataclasses?

- Pure Python dataclass code isn't particularly performant, so isn't fast with large amounts of data
- Dataframes provide a whole algebra to deal with a lot of common transformations: joins, aggregations, pivots, etc. that you get for free

#### Why not write code in a faster statically typed language?

- If you have an existing codebase in Python, converting some parts to use Polars requires little activation energy; existing logic can play with it well enough
- Dataframes in Python are pretty fast, and Python can act as a quick-to-develop-in glue language for the underlying efficient logic
- Dataframes might be the right solution regardless: they provide more than just speed—they provide the right algebraic primitives for aggregating, combining, and transforming data

#### Why Polars?

Static typing only works if the underlying data's shape is immutable, otherwise the type would no longer match the shape. Operations on the shape creating new shapes is the way to go, rather than mutating types/shapes in place, and this is the approach that Polars is designed for. This allows us to make the library a light layer on top of Polars, rather than a significant implementation in its own right

#### How is this meant to work for data science where shapes are so ad-hoc?

In short, it's not: if Polars DataFrames are `dict`s, then Typol's are `TypedDict`s. If you have messy shapes, you don't need this, and you should stay in Polars (just like you should use `dict`s for variable keys). It's easy to go back and forth between Typol and Polars since it's a simple wrapper, so if you have some flexibly shaped code and some that's more rigidly shaped, use Typol and direct Polars appropriately and don't be scared of half-and-halfing it

#### Why Ty?

Ty supports intersection types that makes writing joins a lot less involved, it can construct the joint shape on the fly for you. If other type checkers start supporting this then there's no fundamental reason it can't work with them too. 


### Full example


```python
import typol as tp

class Account(tp.Shape):
    name = tp.dimension(str)
    website = tp.dimension(str)
    account_age = tp.dimension(tp.UINT_8)
    phone = tp.dimension(str)


class Contact(tp.Shape):
    email = tp.dimension(str)
    known_since = tp.dimension(tp.UINT_16)
    phone = tp.dimension(str)

# Let's say I have some account data
accounts = tp.DataFrame(Account, ...)
# Maybe a year has gone past
accounts.with_columns(
    # This is type checked so the `+` operator must be on a number, and the used and produced
    # dimensions must all be in `Account`
    accounts.s.account_age + 1
)

# Let's create contacts out of the name and the website
contacts = accounts.transform(
    Contact,
    # This operation must only use dimensions that are available in `Account`, and must end up at
    # a `Contact` dimension. All expression types are also checked to be `str`. All static checks
    (accounts.s.name.str.to_lowercase() + "@" + accounts.s.website).to(Contact.email),
    # Similar to the above, except with `int`s
    (tp.lit(2026) - accounts.s.account_age).to(Contact.known_since),
    # `phone` is in both shapes so we can leave it alone
)

emails = contacts[Contact.email].to_list()
reveal_type(emails)  # list[str], Contact.known_since would reveal to `list[int]`
print("All emails found:", emails)

class PhoneAddress(tp.Shape):
    number = tp.dimension(str)
    street = tp.dimension(str)

# We have some data about the street addresses of phone lines
phone_addresses = tp.DataFrame(PhoneAddress, ...)

# Now lets join our contacts with the phone address data to work out what home addresses we already know
full_details = contacts.join(
    phone_addresses,
    contacts.s.phone.on(phone_addresses.s.number)
)
reveal_type(full_details)  # tp.DataFrame[Contact & PhoneAddress]

reveal_type(full_details.s.street.is_null())  # Expr[PhoneAddress, _, bool]

# We still need to ask some of our friends for their home address so we can send out RSVPs
still_need_to_ask_for_address = full_details.filter(full_details.s.street.is_null())[full_details.s.email]
reveal_type(still_need_to_ask_for_address)  # tp.Series[str]

# Send out an email asking if they can let us know where to send the RSVPs
send_email(still_need_to_ask_for_address.to_list(), "Send me your mailing address for birthday RSVPs!")
```

More examples and snippets are available [in the `tests`](./tests)