Wie in der Anleitung in der Bachelorarbeit beschrieben, verwende ich für jeden Teil der Bachelorarbeit einen eigenen Workspace. Jeder Ordner enthält die benötigten Daten (Bilder, Modelle) um die zugehörige Fragestellung zu bearbeiten.
Da die Strukturen innerhalb der Workspace-Ordner alle gleich sind, können die Konsolen-Befehle zum Trainieren, Exportieren und Detektieren leicht übernommen werden, indem nur der übergeordnete Pfad angepasst wird. 

Die einzelnen Ordner sind wie folgt zuzuordnen:

- training_demo:	  Dient zur Einführung in die Nutzung der TensorFlow API 
					  Enthält ein Modell aus dem TensorFlow Model Zoo, basiert auf den Trainingsbildern unter "Laborbedinungen" aus dem THGA-Kurs "Softwaretechnik" von Prof. Dr. Hubert Welp, aufgenommen vom Studenten Rami Alkhooli
- model_comparison:	  Arbeitet ebenfalls auf den Bildern aus dem THGA-Kurs und verwendet verschiedene Modelle, um den Einfluss der Modellauswahl zu demonstrieren
- sweets_HD:		  Enthält von mir aufgenommene Bilder, jedoch in hoher Auflösung. Die Bilder entstanden nicht mehr unter idealen Bedingungen hinsichtlich Beleuchtung, Verwendung der immer selben Objekte (wenn auch die gleichen wie im Moodle-Kurs) und Kameraposition (da lediglich ein improvisiertes Stativ zur Verfügung stand). Hintergrund ist ein weißes DIN A3 Blatt, da kein weißer Tisch verfügbar war.
					  Die Bilder wurden auf die Auflösung 1280 x 960 Pixel skaliert. Es werden drei ausgewählte TensorFlow-Modelle verwendet. Wird in der Bachelorarbeit mit sweets_lr verglichen.
- sweets_lr:		  Verwendet die selben Bilder und Modelle wie sweets_HD, diesmal jedoch auf 640 x 480 Pixel skaliert. Wird in der Bachelorarbeit mit sweets_HD verglichen.
- runtime_comparison: Verwendet zwei ausgewählte Modelle und vergleicht die Änderungen der Detektionsergebnisse auf den Bildern aus dem Ordner sweets_HD
- sweets_backgrounds: Enthält Bilder auf verschiedenen Hintergründen und nutzt die selben Modelle wie im Abschnitt sweets_HD 
- tools:              Demonstriert an einem ausgewählten Modell, welches mit den selben Einstellungen wie bei sweets_HD trainiert wurde, dass auch vollkommen andere und kompliziertere Objekte, wie Schrauben, Nägel, Dübel und verschiedene Werkzeuge erkannt und unterschieden werden können.
