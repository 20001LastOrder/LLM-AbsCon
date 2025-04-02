from langchain_core.prompts.chat import ChatPromptTemplate

WORDNET_PROMPT_SIMPLE = """You are an expert constructing a taxonomy from a list of concepts. Given a list of concepts,construct a taxonomy by creating a list of their parent-child relationships. Notice that the taxonomy should respect the following constraints:
1. There should be only one root
2. The root can reach any node in the taxonomy
3. Each node, except root, should have exactly one parent

Make sure the final relations follows the specified format. Let's say we have a concept A and a concept B, represent concept A is a child of concept B using the following format:
A is a child of B

Output the final taxonomy in the following format: 
```taxonomy
<result>
```

Below are some examples
# Examples
Concepts: blast, backfire, explosion, bomb blast, nuclear explosion, inflation, blowback, backblast, airburst, big bang, fragmentation

```taxonomy
airburst is the child of explosion
blowback is the child of explosion
big bang is the child of explosion
inflation is the child of explosion
blast is the child of explosion
backfire is the child of explosion
fragmentation is the child of explosion
backblast is the child of blowback
bomb blast is the child of blast
nuclear explosion is the child of bomb blast
```

Concepts: hand glass, loupe, ultramicroscope, compound microscope, operating microscope, binocular microscope, field-emission microscope, electron microscope, microscope, angioscope, light microscope

```taxonomy
angioscope is the child of microscope
light microscope is the child of microscope
electron microscope is the child of microscope
binocular microscope is the child of light microscope
hand glass is the child of light microscope
compound microscope is the child of light microscope
ultramicroscope is the child of light microscope
operating microscope is the child of binocular microscope
loupe is the child of hand glass
field-emission microscope is the child of electron microscope
```


Concepts: T-junction, interchange, railway junction, spaghetti junction, cloverleaf, intersection, corner, junction, blind corner, level crossing, traffic circle

```taxonomy
intersection is the child of junction
T-junction is the child of junction
interchange is the child of junction
railway junction is the child of junction
traffic circle is the child of junction
corner is the child of intersection
level crossing is the child of intersection
blind corner is the child of corner
cloverleaf is the child of interchange
spaghetti junction is the child of interchange
```

Below is the input to create a taxonomy from:
{user_input}
"""

WORDNET_PROMPT = """You are an expert constructing a taxonomy from a list of concepts. Given a list of concepts,construct a taxonomy by creating a list of their parent-child relationships. Notice that the taxonomy should respect the following constraints:
1. There should be only one root
2. The root can reach any node in the taxonomy
3. Each node, except root, should have exactly one parent

First explain how the taxonomy should be constructed and then output the final taxonomy in the following format: 
```taxonomy
<result>
```
---
Concepts: blast, backfire, explosion, bomb blast, nuclear explosion, inflation, blowback, backblast, airburst, big bang, fragmentation
---
The taxonomy tree starts with "explosion" as the root, which serves as a broad category encompassing various types of explosive events. Subcategories such as "airburst," "blowback," "big bang," "inflation," "blast," "backfire," and "fragmentation" represent different forms or consequences of explosions, each with distinct characteristics. For example, "blowback" is a specific type of explosion involving a reverse or recoil effect, which then further narrows into "backblast," indicating the backward force often associated with this type. The "blast" category includes more destructive and large-scale explosions, with "bomb blast" as a significant subclass, focusing on explosions resulting from bombs. Within this classification, "nuclear explosion" sits as a child of "bomb blast," representing a highly destructive form of bomb-induced blast with nuclear energy release. This hierarchical structure categorizes explosions based on their nature, effects, and causes, providing a logical progression from general types to specific instances.

```taxonomy
airburst is the child of explosion
blowback is the child of explosion
big bang is the child of explosion
inflation is the child of explosion
blast is the child of explosion
backfire is the child of explosion
fragmentation is the child of explosion
backblast is the child of blowback
bomb blast is the child of blast
nuclear explosion is the child of bomb blast
```
---
Concepts: hand glass, loupe, ultramicroscope, compound microscope, operating microscope, binocular microscope, field-emission microscope, electron microscope, microscope, angioscope, light microscope
---

The taxonomy tree here is organized with "microscope" as the root, encapsulating various devices used for magnification and observation of small objects. Directly under this root are specific types, including "angioscope," "light microscope," and "electron microscope," which represent different mechanisms or applications of magnification. The "light microscope" category branches into more specialized forms like "binocular microscope," which has dual eyepieces for enhanced depth perception; "hand glass," a simple magnification tool that leads to the "loupe," often used for close-up inspection; "compound microscope," which has multiple lenses for higher magnification; and "ultramicroscope," designed to view particles smaller than the wavelength of visible light. Under "binocular microscope," the "operating microscope" is a specific type used in surgical procedures, indicating a more specialized application. Meanwhile, the "electron microscope" branch includes the "field-emission microscope," which utilizes a high-resolution field emission for observing extremely small structures. This structure logically classifies microscopes from broad categories to increasingly specialized instruments based on their functionality and design.

```taxonomy
angioscope is the child of microscope
light microscope is the child of microscope
electron microscope is the child of microscope
binocular microscope is the child of light microscope
hand glass is the child of light microscope
compound microscope is the child of light microscope
ultramicroscope is the child of light microscope
operating microscope is the child of binocular microscope
loupe is the child of hand glass
field-emission microscope is the child of electron microscope
```
---
Concepts: T-junction, interchange, railway junction, spaghetti junction, cloverleaf, intersection, corner, junction, blind corner, level crossing, traffic circle
---
The taxonomy tree begins with "junction" as the root, encompassing various types of connections where roads, paths, or railways converge. Direct children of "junction" include "intersection," "T-junction," "interchange," "railway junction," and "traffic circle," each representing a distinct type of convergence. The "intersection" category pertains to crossings where two or more pathways meet, leading to subcategories like "corner," which specifies a meeting point forming an angle, and "level crossing," where a roadway and railway intersect. "Corner" further branches to "blind corner," indicating limited visibility, enhancing the specificity of this classification. The "interchange" category includes complex road structures like "cloverleaf," which provides seamless highway transitions, and "spaghetti junction," a highly intricate series of overlapping and intersecting roads. This hierarchical arrangement logically organizes junction types from general to more specialized forms, based on layout and functionality.

```taxonomy
intersection is the child of junction
T-junction is the child of junction
interchange is the child of junction
railway junction is the child of junction
traffic circle is the child of junction
corner is the child of intersection
level crossing is the child of intersection
blind corner is the child of corner
cloverleaf is the child of interchange
spaghetti junction is the child of interchange
```
"""


CCS_PROMPT = """You are an expert constructing a taxonomy from a list of concepts in computer science. Given a list of concepts,construct a taxonomy by creating a list of their parent-child relationships. Notice that the taxonomy should respect the following constraints:
1. There should be only one root
2. The root can reach any node in the taxonomy

First explain how the taxonomy should be constructed and then output the final taxonomy in the following format: 
```taxonomy
<result>
```
---
Concepts: Solvers, Statistical software, Mathematical software, Mathematical software performance
---
The taxonomy relationships begin with Mathematical software as the root topic because it refers to software designed to solve mathematical problems. Under this, Solvers are a subtopic since they are specialized tools within mathematical software used to solve specific types of mathematical equations or systems. Statistical software is another subtopic because it focuses on applying mathematical principles, particularly probability and statistics, through specialized software to analyze data. Finally, Mathematical software performance is also categorized under Mathematical software because it deals with the efficiency and effectiveness of these programs, addressing how well they perform tasks. Each subtopic represents a more focused area within the overarching field of mathematical software.

```taxonomy
Solvers is a subtopic of Mathematical software
Statistical software is a subtopic of Mathematical software
Mathematical software performance is a subtopic of Mathematical software
```
---
Concepts: Distributed algorithms, MapReduce algorithms, Distributed computing methodologies, Distributed programming languages, Self-organization
---
The taxonomy starts with Distributed computing methodologies as the root, as it is the broadest principles and frameworks for processing tasks across multiple systems or nodes. Distributed algorithms is a subtopic under this, as it deals with the specific techniques used to solve problems in a distributed system. MapReduce algorithms is a subtopic of Distributed algorithms because it represents a specific paradigm for processing large datasets across distributed systems. Self-organization is another subtopic of Distributed algorithms as it refers to systems where distributed components organize themselves without centralized control. Lastly, Distributed programming languages falls under Distributed computing methodologies since it focuses on the languages designed to support the writing of software for distributed systems. Each concept logically narrows down from broad methodologies to specific algorithms and techniques.

```taxonomy
Distributed algorithms is a subtopic of Distributed computing methodologies
MapReduce algorithms is a subtopic of Distributed algorithms
Self-organization is a subtopic of Distributed algorithms
Distributed programming languages is a subtopic of Distributed computing methodologies
```
---
Concepts: Security protocols, Web protocol security, Network security, Firewalls, Mobile and wireless security, Denial-of-service attacks
---
The taxonomy begins with Network security as the root, covering the broad spectrum of measures and techniques used to protect data and resources in networked environments. Security protocols is a subtopic of Network security because these protocols define the rules and standards for securing data transmission over a network. Web protocol security is another subtopic, focusing specifically on securing web-based communications and transactions, such as HTTPS, under the broader network security umbrella. Mobile and wireless security is a subtopic due to the unique challenges in securing mobile devices and wireless communication channels, requiring specialized solutions within the network security domain. Denial-of-service (DoS) attacks is also categorized under Network security, as it is a specific threat type aimed at disrupting network services. Finally, Firewalls are a subtopic because they represent one of the key defensive technologies used to secure networks by controlling incoming and outgoing traffic based on predetermined security rules. Each subtopic focuses on a particular aspect or technology essential to protecting network infrastructure.

```taxonomy
Security protocols is a subtopic of Network security
Web protocol security is a subtopic of Network security
Mobile and wireless security is a subtopic of Network security
Denial-of-service attacks is a subtopic of Network security
Firewalls is a subtopic of Network security
```
"""


def get_relation(dataset_name):
    if dataset_name == "wordnet":
        relation = "is the child of"
    elif dataset_name == "ccs":
        relation = "is a subtopic of"
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    return relation


def get_prompt(dataset_name, prompt_method="fewshot") -> ChatPromptTemplate:
    if prompt_method == "simple":
        return ChatPromptTemplate([("human", WORDNET_PROMPT_SIMPLE)])

    if dataset_name == "wordnet":
        prompt = WORDNET_PROMPT
    elif dataset_name == "ccs":
        prompt = CCS_PROMPT
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")
    paragraphs = prompt.split("---")
    messages = [("system", paragraphs[0])]
    for i, paragraph in enumerate(paragraphs[1:]):
        if i % 2 == 0:
            messages.append(("user", paragraph))
        else:
            messages.append(("assistant", paragraph))
    messages.append(("human", " {user_input}"))

    return ChatPromptTemplate(messages)
