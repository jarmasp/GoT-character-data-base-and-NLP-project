Exploratory data analysing using the book series A Song of Ice and Fire

1. Aims, objectives and background

1.1. Introduction

A Song of Ice of and Fire it's a book series writen by the Author George R. R. Martin, the series was made into a television show and it exploded in popularity.

Having recently heard the rumor a the next book in the series Winds of Winter might be releasing this winter I wanted to recapitulate the events by looking at which character has one connection with another, and the factions each character is a part of. 

1.2. Aims and objectives
Within this project, I would like to explore the following:

Get to know Spacy.

Analyze text data and observe if the change of importance in any characters can predict if the same character will have POV chapter in the following book
Does a change of importance can predict if the character will die?
What communities exists in ASOIF?
Who is part of what community?
Do changes in the communities occur? i.e. a character changing from one community to another.

1.3. Steps of the project

Create a list of every named character in the book series.
Use Spacy NLP to read the books and identify the characters.
Count the number of interactions a character has had with another.
Determine the different communities each character is a part of.
Determine the importance of each character by they degrees of centrality in a relationship network.
Create a visualization of the relationship network.
Create a graph to visualize the changes importance of the POV character over the course of the books.

1.4. Dataset

Data selection

The character data was created by scrapping the Ice and Fire fandom wiki (https://iceandfire.fandom.com/wiki/A_Song_of_Ice_and_Fire_Wiki) character page, the entity list was created by using Spacy to read the ASOIF books on .txt format to identify the entities, them using the scrapped character list, the entity was cleaned to only contain the named characters in the book series

Data limitations

This data set has the issue that much as in the real world in the ASOIF universe some names are repeated, the are several targaryen with the name Aegon for example, this results in different relationships being inflated because the software can't recognize context clues to know if the Aegon the character is talking about it's Aegon the conqueror or Aegon the VI. The same is true in the several cases of characters using fake names
there is also the issue with the tittles, the setting being in a feudal society the characters refer to one another by their tittle like ser, my lord, My king, your highness and because of that several interacctions are lost.
Another special case to mention is Daenerys, one of the main characters it's usual mentioned by pseudonym 'Misha' or one of her many tittles which are not accounted for.
