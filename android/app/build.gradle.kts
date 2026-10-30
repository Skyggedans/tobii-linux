plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.tobiilinux.poc"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.tobiilinux.poc"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // The .so (built separately with `cargo ndk`) is checked in under
    // src/main/jniLibs/<abi>/ — the Android Gradle Plugin picks that up
    // without further configuration. No NDK/CMake build here.
    ndkVersion = "25.1.8937393"
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
}
