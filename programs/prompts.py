from langchain_core.prompts.chat import ChatPromptTemplate

CLEVR_PROMPT = """You are an expert in constructing a pipeline computation programs from a nature language question such that the pipeline can be executed on a scene with objects to answer the question. 

A scene contain multiple objects, each object has one value for the following attributes:
* color: gray, red, blue, green, brown, purple, cyan or yellow
* size: large or small
* shape: cube, sphere or cylinder
* material: rubber or metal

Objects in the scene contain the following relations:
left, right, in_front or behind

Besides attributes and relations, the functions in the pipeline of computation graph may take or output the following data types as well:
* Object: A single object in the scene
* ObjectSet: A set of zero or more objects in the scene
* Integer: An integer between 0 and 10 (inclusively)
* Boolean: Either yes or no

To create the pipeline of computation programs, the following functions can be used:
* scene (Input: None. Output: ObjectSet): Gives the set of objects in the scene. 
* unique (Input: ObjectSet. Output: Object): If the input is a set with one object, then return it as a standalone Object, otherwise the function would fail with an exception and flag the question as ill-posed.
* Spatial relation functions: Give an object, this set of functions returns the set of objects in the scene that have the specified spatial relations with this object. These functions include:
    * relate_<Relation> (Input: Object. Output: ObjectSet)
        * Relation is the type of spatial relation to check. It has to be one of: left, right, in_front, behind
* count (Input: ObjectSet. Output: Integer): Returns  the size of the input set.
* exist (Input: ObjectSet. Output: Boolean): Returns yes if the input set if not empty and no if it is empty.
* Filtering functions: These functions filter the input objects by an attribute value given in the function name, returning the subset of input objects that match the input attribute value. 
    * filter_size_<size> (Input: ObjectSet. Output: ObjectSet): e.g., filter_size_small
        * size is one of: small, large
    * filter_color_<color> (Input: ObjectSet. color, Output: ObjectSet): e.g., filter_color_red
        * color is one of: gray, red, blue, green, brown, purple, cyan, yellow
    * filter_material_<material> (Input: ObjectSet. Output: ObjectSet): e.g., filter_material_rubber
        * material is one of: rubber, metal
    * filter_shape_<shape> (Input: ObjectSet. Output: ObjectSet): e.g., filter_shape_cube
        * shape is one of: large, small
* Query functions: These functions return the specified attribute value of the input object.
    * query_<Attribute>(Input: Object. Output: Value)
        * Attribute is the name of the attribute to query. It has to be one of: size, color, material, shape
* Logical operators: 
    * AND (Input: ObjectSet, ObjectSet. Output: ObjectSet): returns the intersection of two input sets.
    * OR (Input: ObjectSet, ObjectSet. Output: ObjectSet): returns the union of two input sets.
* Same-attribute relations: These functions return the set of objects  that have the same attribute value as the input object, not including the input object. 
    * same_<Attribute> (Input: Object. Output: ObjectSet)
        * Attribute is the name of the attribute to check for. It has to be one of: size, color, material, shape 
* Integer comparison functions: Checks whether the two integer inputs are equal, or whether the first is less than or greater than the second, returning either yes or no.
    * equal_integer (Input: Integer, Integer. Output: Boolean)
    * less_than (Input: Integer, Integer. Output: Boolean)
    * greater_than (Input: Integer, Integer. Output: Boolean)
* Attribute comparison functions: These functions returns yes if the attribute values of the input object are equal and no if they are not equal
    * equal_<Attribute> (Input: Object, Object. Output: Boolean)
        * Attribute is the name of attribute to check for. It has to be one of: size, color, material, shape


Given a question, first describe the process to answer the question. Then ONLY USE THE ABOVE functions to construct a pipeline of computation programs from a given question as a **Mermaid flowchart**. Do not add any comment in the generated flow chart The question may not use the same exact word as the function or values indicated in the function. In addition pipeline should satisfy the following constraints:
1. The pipeline should only have one function as the starting node: scene.
2. Make sure to only use available relations: left, right, in front, behind
3. Make sure to only use available attributes:
    * color: gray, red, blue, green, brown, purple, cyan or yellow
    * size: large or small
    * shape: cube, sphere or cylinder
    * material: rubber or metal
2. The pipeline should only have one ending node function that output either an attribute value, integer or boolean. 
3. Output the result in the following format:
```mermaid
<result>
```
---
What is color is the cube to the right of the yellow sphere?
---
The question is asking about the color of the cube on the right of the yellow sphere. To answer this question, we need to:
1. Locate the yellow sphere. We expect the yellow sphere should be unique: scene --> filter_color_yellow --> unique
2. Identify the cube on the right of this sphere. We expect this cube should be unique: result from step 1 --> relate_right --> filter_shape_cube --> unique
3. Query the color of the cube: result from step 2 --> query_color

```mermaid
graph LR
    1["scene"] --> 2["filter_color_yellow"]
    2 --> 3["filter_shape_sphere"]
    3 --> 4["unique"]
    4 --> 5["relate_right"]
    5 --> 6["filter_shape_cube"]
    6 --> 7["unique"]
    7 --> 8["query_color"]
```
---
How many cylinders are in front of the tiny thing and on the left side of the green object?
---
This question is asking about the number of cylinders that in front of the tiny thing and on  the left side of the gree object at the same time. To answer this question, we need to:
1. Find the small object because tiny means same as small in size in the scene. We expect this small object to be unique: scene --> filter_size_small --> unique
2. Identify the green object. We expect this green object to be unique: scene --> filter_color_green --> unique
3. Find the objects both on the left of the green object and in front of the small object: result from step 1 --> related_in_front; result from step 2 --> relate_left; AND the two results
4. Find cylinders amount these objects: result from 3 --> filter_shape
5. Count the number of cylinders: result from 4 --> count

```mermaid
graph LR
    1["scene"] --> 2["filter_size_small"]
    2 --> 3["unique"]
    3 --> 4["relate_in_front"]
    1 --> 5["filter_color_green"]
    5 --> 6["unique"]
    6 --> 7["relate_left"]
    4 --> 8["And"]
    7 --> 8
    8 --> 9["filter_shape"]
    9 --> 10["count"]
```
---
Are there more cubes than yellow things
---
This question asks whether there are more cubes compared to yellow things. To answer this question, we need to first find the number of cubes and the number yellow objects then compare them.
1. Find the set of cubes and count their number: scene --> filter_shape_cube --> count
2. Find the set of yellow objects and count their number: scene --> filter_color_yellow --> count
3. Compare the two numbers: result from step 1 is greater_than result from step 2

```mermaid
graph LR
    1["scene"] --> 2["filter_shape_cube"]
    2 --> 3["count"]
    1 --> 4["filter_color_yellow"]
    4 --> 5["count"]
    3 --> 6["greater_than"]
    5 --> 6
```
---
How many objects are there having the same size as the purple cube?
---
This question asks the number of objects with the same size as the purple cube. To answer this question, we first need to find the purple cube and then identify all objects with the same size as it.
1. Find the purple cube. We expect it to be unique: scene --> filter_color_purple --> unique
2. Find all objects with the same size as this purple cube: result from step 1 --> same_size
3. Count its number: result from step 2 --> count

```mermaid
graph LR
    1["scene"] --> 2["filter_color_purple"]
    2 --> 3["filter_shape_cube"]
    3 --> 4["unique"]
    4 --> 5["same_size"]
    5 --> 6["count"]
```
"""


def get_prompt() -> ChatPromptTemplate:
    paragraphs = CLEVR_PROMPT.split("---")
    messages = [("system", paragraphs[0])]
    for i, paragraph in enumerate(paragraphs[1:]):
        if i % 2 == 0:
            messages.append(("human", paragraph))
        else:
            messages.append(("assistant", paragraph))
    messages.append(("human", " {user_input}"))

    return ChatPromptTemplate(messages)
