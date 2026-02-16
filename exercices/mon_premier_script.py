message = "C'est mon premier script !!!"
print(message)


je_change_de_type = 1
print(type(je_change_de_type)) ##int
je_change_de_type = "coucou"
print(type(je_change_de_type)) ##str

prenoms = ["Guillaume", "Gilles", "Juliette", "Antoine", "François", "Cassandre"]
more_than_seven = 0
for prenom in prenoms:
    if len(prenom) > 7:
        more_than_seven += 1
        print(prenom + " est un prénom avec un nombre de lettres supérieur à 7")
    else:
        print(prenom + " est un prénom avec un nombre de lettres inférieur ou égal à 7")
print("Nombre de prénoms dont le nombre de lettres est supérieur à 7 : " + str(more_than_seven))

def saluer(nom: str) -> str:
    return "Bonjour " + nom

print(saluer("Alice"))  # Affiche : Bonjour Alice