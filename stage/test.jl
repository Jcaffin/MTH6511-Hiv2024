function encode_message(message::String, shift::Int)::String
    encoded_message = ""
    for char in message
        if 'a' <= char <= 'z'
            shifted_char = Char((Int(char) - Int('a') + shift) % 26 + Int('a'))
            encoded_message *= shifted_char
        elseif 'A' <= char <= 'Z'
            shifted_char = Char((Int(char) - Int('A') + shift) % 26 + Int('A'))
            encoded_message *= shifted_char
        else
            encoded_message *= char
        end
    end
    return encoded_message
end

# Utilisation de la fonction pour encoder "salut" avec un décalage de 10 (a -> k)
message = "solution : bien vu aaa"
shift = 10
encoded_message = encode_message(message, shift)
println("Message encodé : $encoded_message")
