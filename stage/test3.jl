using Luxor
using Plots

# Créer un plot simple
x = 1:10
y = rand(10)
p = plot(x, y, title="Mon plot", label="Données", xlabel="X", ylabel="Y")

# Enregistrer le plot en tant qu'image temporaire
temp_image_path = "temp_plot.png"
savefig(p, temp_image_path)

# Définir la fonction pour créer le contenu du PDF
function create_pdf()
    # Définir la taille de la police
    fontsize(20)

    # Ajouter du texte sur le PDF
    Luxor.text("Voici du texte à enregistrer.", Point(100, 750), halign=:left)

    # Lire l'image temporaire et placer dans le PDF
    img = readpng(temp_image_path)
    placeimage(img, Point(300, 400), 1.0)
end

# Appel de la macro @pdf pour générer le PDF avec le contenu créé
@pdf "document.pdf" 600 800 create_pdf()

# Supprimer l'image temporaire
rm(temp_image_path)
