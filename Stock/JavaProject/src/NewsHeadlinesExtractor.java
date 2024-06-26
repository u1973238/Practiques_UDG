import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NewsHeadlinesExtractor {

    public static void main(String[] args) {
        String url = "https://markets.businessinsider.com/cryptocurrencies"; // URL real de noticias financieras
        String outputFilePath = "headlines.txt"; // Ruta del archivo de salida
        try {
            // Conectar al sitio web y obtener el documento
            Document document = Jsoup.connect(url).get();

            // Seleccionar los elementos que contienen los titulares de noticias
            // Reemplaza .cd__headline-text con la clase real encontrada en la inspección de la página
            Elements headlineElements = document.select(".top-story__link");

            // Extraer el texto de cada elemento y agregarlo a la lista de titulares
            List<String> headlines = new ArrayList<>();
            for (Element element : headlineElements) {
                headlines.add(element.text());
            }

            // Guardar los titulares en un archivo de texto
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
                for (String headline : headlines) {
                    writer.write(headline);
                    writer.newLine();
                }
                System.out.println("Los titulares se han guardado en " + outputFilePath);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
