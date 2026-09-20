import org.jetbrains.intellij.platform.gradle.TestFrameworkType

plugins {
    java
    id("org.jetbrains.intellij.platform")
}

val productVersion = Regex("""(?m)^version\s*=\s*"([^"]+)"""")
    .find(file("../../pyproject.toml").readText())
    ?.groupValues
    ?.get(1)
    ?: error("Unable to read Entroly version from ../../pyproject.toml")

val wrapperVersion = Regex("""gradle-(.+)-(?:bin|all)\.zip""")
    .find(file("gradle/wrapper/gradle-wrapper.properties").readText())
    ?.groupValues
    ?.get(1)
    ?: error("Unable to read the Gradle version from gradle/wrapper/gradle-wrapper.properties")

group = "io.github.juyterman1000"
version = productVersion

dependencies {
    intellijPlatform {
        intellijIdeaCommunity("2024.3.6")
        testFramework(TestFrameworkType.Platform)
    }

    testImplementation("junit:junit:4.13.2")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

// The plugin version is a build-time fact, already flowing from pyproject.toml
// into plugin.xml. Reading it back at runtime means depending on the platform's
// descriptor APIs, and those keep being reclassified: PluginManagerCore.getPlugin
// is @ApiStatus.Internal, PluginManager.getPlugin(PluginId) is a deprecated
// delegate to it, and PluginManager.getPluginByClass became internal between
// builds 261 and 262. Baking the value in depends on no platform API at all.
val generateVersionConstant by tasks.registering {
    val outputDir = layout.buildDirectory.dir("generated/sources/entrolyVersion/java")
    val value = productVersion
    inputs.property("entrolyVersion", value)
    outputs.dir(outputDir)
    doLast {
        val packageDir = outputDir.get().asFile.resolve("io/github/juyterman1000/entroly")
        packageDir.mkdirs()
        packageDir.resolve("EntrolyVersion.java").writeText(
            """
            package io.github.juyterman1000.entroly;

            /** Generated from pyproject.toml at build time. Do not edit. */
            final class EntrolyVersion {
                static final String VALUE = "$value";

                private EntrolyVersion() {
                }
            }
            """.trimIndent() + "\n"
        )
    }
}

sourceSets {
    main {
        java.srcDir(generateVersionConstant)
    }
}

intellijPlatform {
    pluginConfiguration {
        id = "io.github.juyterman1000.entroly"
        name = "Entroly"
        version = productVersion

        ideaVersion {
            sinceBuild = "243"
            untilBuild = provider { null }
        }

        vendor {
            name = "Entroly"
            email = "fastrunner10090@gmail.com"
            url = "https://github.com/juyterman1000/entroly"
        }
    }

    pluginVerification {
        ides {
            recommended()
        }
    }

    publishing {
        token = providers.environmentVariable("PUBLISH_TOKEN")
    }
}

tasks {
    test {
        useJUnit()
    }

    wrapper {
        // gradle/wrapper/gradle-wrapper.properties is the single source of truth
        // and is what dependabot bumps. Restating the version here would make
        // `./gradlew wrapper` regenerate the wrapper at the stale value and
        // silently revert that bump, with both files still parsing fine.
        gradleVersion = wrapperVersion
        distributionType = Wrapper.DistributionType.BIN
    }
}
