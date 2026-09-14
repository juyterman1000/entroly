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
        gradleVersion = "9.5.0"
        distributionType = Wrapper.DistributionType.BIN
    }
}
