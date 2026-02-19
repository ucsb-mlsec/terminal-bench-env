package com.example;

import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.annotation.processing.SupportedSourceVersion;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.tools.JavaFileObject;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Set;

@SupportedAnnotationTypes("com.example.AutoProperty")
@SupportedSourceVersion(SourceVersion.RELEASE_8)
public class AutoPropertyProcessor extends AbstractProcessor {

    @Override
    public Set<String> getSupportedAnnotationTypes() {
        return Set.of("com.example.AutoProperty");
    }

    @Override
    public SourceVersion getSupportedSourceVersion() {
        return SourceVersion.latestSupported();
    }

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
        for (Element element : roundEnv.getElementsAnnotatedWith(AutoProperty.class)) {
            if (element.getKind() == ElementKind.CLASS) {
                TypeElement classElement = (TypeElement) element;
                try {
                    generateCode(classElement);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return true;
    }

    private void generateCode(TypeElement classElement) throws IOException {
        String packageName = processingEnv.getElementUtils().getPackageOf(classElement).getQualifiedName().toString();
        String className = classElement.getSimpleName().toString();
        String generatedClassName = className + "Generated";
        String qualifiedGeneratedClassName = packageName.isEmpty() ? generatedClassName : packageName + "." + generatedClassName;

        JavaFileObject fileObject = processingEnv.getFiler().createSourceFile(qualifiedGeneratedClassName);
        
        try (PrintWriter writer = new PrintWriter(fileObject.openWriter())) {
            if (!packageName.isEmpty()) {
                writer.println("package " + packageName + ";");
                writer.println();
            }

            writer.println("public class " + generatedClassName + " extends " + className + " {");
            writer.println();

            for (Element enclosedElement : classElement.getEnclosedElements()) {
                if (enclosedElement.getKind() == ElementKind.FIELD) {
                    VariableElement fieldElement = (VariableElement) enclosedElement;
                    if (fieldElement.getModifiers().contains(Modifier.PRIVATE)) {
                        String fieldName = fieldElement.getSimpleName().toString();
                        String fieldType = fieldElement.asType().toString();
                        String capitalizedFieldName = capitalize(fieldName);

                        writer.println("    public " + fieldType + " get" + capitalizedFieldName + "() {");
                        writer.println("        return this." + fieldName + ";");
                        writer.println("    }");
                        writer.println();

                        writer.println("    public void set" + capitalizedFieldName + "(" + fieldType + " " + fieldName + ") {");
                        writer.println("        this." + fieldName + " = " + fieldName + ";");
                        writer.println("    }");
                        writer.println();
                    }
                }
            }

            writer.println("}");
        }
    }

    private String capitalize(String str) {
        if (str == null || str.isEmpty()) {
            return str;
        }
        return str.substring(0, 1).toUpperCase() + str.substring(1);
    }
}