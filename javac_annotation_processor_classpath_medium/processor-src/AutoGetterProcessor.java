package com.example.processor;

import com.example.annotations.AutoGetter;

import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.annotation.processing.SupportedSourceVersion;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.element.Modifier;
import javax.tools.JavaFileObject;
import java.io.IOException;
import java.io.Writer;
import java.util.Set;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

@SupportedAnnotationTypes("com.example.annotations.AutoGetter")
@SupportedSourceVersion(SourceVersion.RELEASE_8)
public class AutoGetterProcessor extends AbstractProcessor {

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
        if (annotations.isEmpty()) {
            return false;
        }

        Map<TypeElement, List<VariableElement>> classFieldsMap = new HashMap<>();

        for (Element element : roundEnv.getElementsAnnotatedWith(AutoGetter.class)) {
            if (element.getKind() == ElementKind.FIELD) {
                VariableElement field = (VariableElement) element;
                TypeElement classElement = (TypeElement) field.getEnclosingElement();
                
                classFieldsMap.computeIfAbsent(classElement, k -> new ArrayList<>()).add(field);
            }
        }

        for (Map.Entry<TypeElement, List<VariableElement>> entry : classFieldsMap.entrySet()) {
            try {
                generateGetterClass(entry.getKey(), entry.getValue());
            } catch (IOException e) {
                processingEnv.getMessager().printMessage(
                    javax.tools.Diagnostic.Kind.ERROR,
                    "Failed to generate getter class: " + e.getMessage()
                );
            }
        }

        return true;
    }

    private void generateGetterClass(TypeElement classElement, List<VariableElement> fields) throws IOException {
        String packageName = processingEnv.getElementUtils().getPackageOf(classElement).getQualifiedName().toString();
        String className = classElement.getSimpleName().toString();
        String generatedClassName = className + "Generated";
        String qualifiedGeneratedClassName = packageName.isEmpty() ? generatedClassName : packageName + "." + generatedClassName;

        JavaFileObject fileObject = processingEnv.getFiler().createSourceFile(qualifiedGeneratedClassName);
        
        try (Writer writer = fileObject.openWriter()) {
            if (!packageName.isEmpty()) {
                writer.write("package " + packageName + ";\n\n");
            }

            writer.write("public class " + generatedClassName + " extends " + className + " {\n\n");

            for (VariableElement field : fields) {
                String fieldName = field.getSimpleName().toString();
                String fieldType = field.asType().toString();
                String getterName = "get" + capitalize(fieldName);

                writer.write("    public " + fieldType + " " + getterName + "() {\n");
                writer.write("        return this." + fieldName + ";\n");
                writer.write("    }\n\n");
            }

            writer.write("}\n");
        }
    }

    private String capitalize(String str) {
        if (str == null || str.isEmpty()) {
            return str;
        }
        return Character.toUpperCase(str.charAt(0)) + str.substring(1);
    }
}