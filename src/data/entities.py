"""Two-word entity lists for the bimodal prompt bank (prereg §4.1).

Exploration categories (the smoke categories): us_city, country, scientist.
Confirmation categories: author, athlete, company, dish, film, musician, landmark,
fictional_character, painter.
All entries: exactly two whitespace-separated words, ASCII letters only (no hyphens or apostrophes).
"""

CATEGORIES = {
    # ---------------- exploration ----------------
    "us_city": ("US city", [
        "New York", "Los Angeles", "San Francisco", "San Diego", "Las Vegas", "New Orleans",
        "San Antonio", "San Jose", "Santa Fe", "Kansas City", "Long Beach", "Fort Worth", "El Paso",
        "Oklahoma City", "Virginia Beach", "Colorado Springs", "Baton Rouge", "Des Moines",
        "Grand Rapids", "Little Rock", "Saint Paul", "Santa Barbara", "Santa Monica", "Palo Alto",
        "Ann Arbor", "Corpus Christi", "Jersey City", "Sioux Falls", "Fort Lauderdale", "Santa Cruz",
        "San Bernardino", "Rapid City", "Carson City", "Atlantic City", "Palm Springs", "Key West",
        "Beverly Hills", "Boca Raton", "Fort Myers", "Iowa City"]),
    "country": ("country", [
        "New Zealand", "South Africa", "South Korea", "North Korea", "Saudi Arabia", "Costa Rica",
        "Sri Lanka", "El Salvador", "United States", "United Kingdom", "Sierra Leone", "Burkina Faso",
        "Cape Verde", "San Marino", "East Timor", "South Sudan", "North Macedonia", "Ivory Coast",
        "Czech Republic", "Dominican Republic", "Equatorial Guinea", "Marshall Islands",
        "Solomon Islands", "Saint Lucia", "Vatican City"]),
    "scientist": ("scientist", [
        "Albert Einstein", "Isaac Newton", "Marie Curie", "Charles Darwin", "Niels Bohr", "Alan Turing",
        "Ada Lovelace", "Galileo Galilei", "Nikola Tesla", "Michael Faraday", "Max Planck",
        "Richard Feynman", "Stephen Hawking", "Rosalind Franklin", "Louis Pasteur", "Gregor Mendel",
        "Johannes Kepler", "Werner Heisenberg", "Enrico Fermi", "Carl Sagan", "Jane Goodall",
        "Edwin Hubble", "Dmitri Mendeleev", "Robert Hooke", "Tycho Brahe", "Paul Dirac", "Lise Meitner",
        "Alexander Fleming", "Edward Jenner", "Barbara McClintock", "Linus Pauling", "Rachel Carson",
        "John Dalton", "Ernest Rutherford", "Thomas Edison", "Leonhard Euler", "Emmy Noether",
        "Claude Shannon", "John Nash"]),
    # ---------------- confirmation ----------------
    "author": ("author", [
        "William Shakespeare", "Jane Austen", "Charles Dickens", "Mark Twain", "Ernest Hemingway",
        "Virginia Woolf", "George Orwell", "Leo Tolstoy", "Franz Kafka", "James Joyce", "Toni Morrison",
        "Agatha Christie", "Stephen King", "Emily Dickinson", "Oscar Wilde", "Herman Melville",
        "Fyodor Dostoevsky", "Victor Hugo", "Mary Shelley", "Charlotte Bronte", "Harper Lee",
        "John Steinbeck", "Kurt Vonnegut", "Sylvia Plath", "Walt Whitman", "Chinua Achebe",
        "Haruki Murakami", "Margaret Atwood", "Zadie Smith", "Salman Rushdie", "Kazuo Ishiguro",
        "Isabel Allende", "Jules Verne", "Lewis Carroll", "Roald Dahl", "Neil Gaiman",
        "Cormac McCarthy", "Ray Bradbury", "Aldous Huxley", "Thomas Hardy"]),
    "athlete": ("athlete", [
        "Michael Jordan", "Serena Williams", "Lionel Messi", "Cristiano Ronaldo", "Usain Bolt",
        "Roger Federer", "Tiger Woods", "Simone Biles", "LeBron James", "Muhammad Ali", "Babe Ruth",
        "Wayne Gretzky", "Michael Phelps", "Rafael Nadal", "Novak Djokovic", "Tom Brady",
        "Kobe Bryant", "Stephen Curry", "Diego Maradona", "Venus Williams", "Carl Lewis", "Jesse Owens",
        "Mia Hamm", "Lewis Hamilton", "Michael Schumacher", "Ayrton Senna", "Magic Johnson",
        "Kevin Durant", "Zinedine Zidane", "Kylian Mbappe", "Steffi Graf", "Andre Agassi",
        "Pete Sampras", "Martina Navratilova", "Katie Ledecky", "Mike Tyson"]),
    "company": ("company", [
        "General Electric", "General Motors", "Goldman Sachs", "Morgan Stanley", "American Express",
        "Home Depot", "Best Buy", "Walt Disney", "Burger King", "Taco Bell", "Pizza Hut",
        "Dunkin Donuts", "Under Armour", "Estee Lauder", "Wells Fargo", "Southwest Airlines",
        "United Airlines", "American Airlines", "Boston Dynamics", "Hewlett Packard",
        "Texas Instruments", "Lockheed Martin", "Northrop Grumman", "Kraft Heinz", "Electronic Arts",
        "Red Bull", "Ralph Lauren", "Calvin Klein", "Hugo Boss", "Krispy Kreme", "Whole Foods",
        "General Mills", "Western Union", "Charles Schwab", "Capital One", "Deutsche Bank",
        "Aston Martin", "Land Rover", "Alfa Romeo"]),
    "dish": ("dish", [
        "Caesar Salad", "Pad Thai", "Fried Rice", "Apple Pie", "Beef Stew", "Chicken Curry",
        "French Toast", "Banana Bread", "Club Sandwich", "Hot Dog", "Clam Chowder", "Greek Salad",
        "Pot Roast", "Egg Rolls", "Spring Rolls", "Pulled Pork", "Fried Chicken", "Butter Chicken",
        "Beef Wellington", "Chicken Parmesan", "Lobster Bisque", "Tomato Soup", "Potato Salad",
        "Carrot Cake", "Lemon Tart", "Apple Crumble", "Rice Pudding", "Fish Pie", "Lamb Kebab",
        "Beef Tacos", "Miso Soup", "Chicken Satay", "Duck Confit", "Tuna Salad", "Garlic Bread"]),
    "film": ("film", [
        "Star Wars", "Jurassic Park", "Pulp Fiction", "Toy Story", "Black Panther", "Iron Man",
        "Top Gun", "Blade Runner", "Forrest Gump", "Fight Club", "Finding Nemo", "Inside Out",
        "Wonder Woman", "Spirited Away", "Citizen Kane", "Mad Max", "Home Alone", "Die Hard",
        "Pretty Woman", "Mean Girls", "Get Out", "Dirty Dancing", "Taxi Driver", "Apocalypse Now",
        "Raging Bull", "Rear Window", "Groundhog Day", "Office Space", "Donnie Darko",
        "Shutter Island", "Gone Girl", "Blue Velvet", "Black Swan", "Pacific Rim"]),
    "musician": ("musician", [
        "Taylor Swift", "Bob Dylan", "Elton John", "Michael Jackson", "Freddie Mercury", "David Bowie",
        "Whitney Houston", "Aretha Franklin", "Elvis Presley", "Johnny Cash", "Billie Eilish",
        "Ed Sheeran", "Bruno Mars", "Lady Gaga", "Kanye West", "Frank Sinatra", "Ray Charles",
        "Stevie Wonder", "Bob Marley", "Jimi Hendrix", "Kurt Cobain", "Amy Winehouse", "Dolly Parton",
        "John Lennon", "Paul McCartney", "Mick Jagger", "Janis Joplin", "Ella Fitzgerald",
        "Miles Davis", "Louis Armstrong", "Nina Simone", "Ariana Grande", "Justin Bieber",
        "Katy Perry", "Britney Spears", "Celine Dion", "Shania Twain", "Bruce Springsteen",
        "Tina Turner"]),
    "landmark": ("landmark", [
        "Eiffel Tower", "Big Ben", "Taj Mahal", "Machu Picchu", "Mount Everest", "Grand Canyon",
        "Niagara Falls", "Mount Rushmore", "Hoover Dam", "Times Square", "Central Park",
        "Buckingham Palace", "Red Square", "Angkor Wat", "Chichen Itza", "Mount Fuji",
        "Victoria Falls", "Table Mountain", "Sagrada Familia", "Notre Dame", "Hagia Sophia",
        "Brandenburg Gate", "Forbidden City", "Great Wall", "Lincoln Memorial", "Space Needle",
        "Tower Bridge", "Westminster Abbey", "Mount Kilimanjaro", "Lake Tahoe", "Death Valley",
        "Burj Khalifa", "Golden Temple", "Blue Mosque", "Mount Olympus"]),
    "fictional_character": ("fictional character", [
        "Harry Potter", "Sherlock Holmes", "Mickey Mouse", "Darth Vader", "Luke Skywalker",
        "James Bond", "Peter Pan", "Bugs Bunny", "Homer Simpson", "Bart Simpson", "Tony Stark",
        "Bruce Wayne", "Clark Kent", "Peter Parker", "Frodo Baggins", "Bilbo Baggins", "Lara Croft",
        "Indiana Jones", "Jack Sparrow", "Hermione Granger", "Ron Weasley", "Atticus Finch",
        "Holden Caulfield", "Jay Gatsby", "Elizabeth Bennet", "Captain America", "Donald Duck",
        "Daffy Duck", "Han Solo", "Lisa Simpson", "Walter White", "Jon Snow", "Arya Stark",
        "Ellen Ripley"]),
    "painter": ("painter", [
        "Claude Monet", "Pablo Picasso", "Salvador Dali", "Frida Kahlo", "Andy Warhol",
        "Jackson Pollock", "Edvard Munch", "Henri Matisse", "Paul Cezanne", "Gustav Klimt",
        "Johannes Vermeer", "Edgar Degas", "Rene Magritte", "Mark Rothko", "Francis Bacon",
        "Paul Gauguin", "Joan Miro", "Wassily Kandinsky", "Piet Mondrian", "Edward Hopper",
        "Diego Rivera", "Paul Klee", "Marc Chagall", "Keith Haring", "Roy Lichtenstein",
        "Egon Schiele", "Camille Pissarro", "Hieronymus Bosch", "Albrecht Durer", "El Greco",
        "Francisco Goya", "Diego Velazquez", "Amedeo Modigliani", "Henri Rousseau",
        "Gerhard Richter", "David Hockney"]),
}

EXPLORATION = ["us_city", "country", "scientist"]
CONFIRMATION = [c for c in CATEGORIES if c not in EXPLORATION]

# Templates. Exploration = the 5 smoke templates (3 open + 2 pair) + 1 new pair template.
OPEN_TEMPLATES_EXPL = [
    "Name one {cat} with a two-word name. Reply with only the name.",
    "Give me a random example of a {cat} whose name has two words. Just the name, nothing else.",
    "Quick: a {cat} with a two-word name? Answer with the name only.",
]
PAIR_TEMPLATES_EXPL = [
    "Pick one of these at random: {a} or {b}. Reply with only your choice.",   # smoke
    "Choose either {a} or {b}. Answer with just the name.",                     # smoke
    "Between {a} and {b}, which one would you pick? Reply with the name only.",  # new
]
PAIR_TEMPLATES_CONF = [
    "Select one: {a} or {b}. Respond with just that name.",
    "I'm deciding between {a} and {b}. Pick one for me and reply with only the name.",
    "Your choice, {a} or {b}? Answer with the name and nothing else.",
    "Flip a mental coin between {a} and {b} and write only the winner's name.",
    "Which do you prefer, {a} or {b}? Give only the name.",
    "Choose one of the following: {a}, {b}. Output just the chosen name.",
    "Of {a} and {b}, name the one you'd go with. Only the name, please.",
    "Respond with exactly one of these names: {a} or {b}.",
]
