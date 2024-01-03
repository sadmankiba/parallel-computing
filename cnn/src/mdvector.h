#pragma once

enum MDVectorTag {
    V1,
    V2,
    V3,
    V4
};

struct MDVector {
    MDVectorTag tag;
    V1D v1;
    V2D v2;
    V3D v3;
    V4D v4;

    MDVector() : tag(MDVectorTag::V1) {}
    MDVector(const V1D& v1) : tag(MDVectorTag::V1), v1(v1) {}
    MDVector(const V2D& v2) : tag(MDVectorTag::V2), v2(v2) {}
    MDVector(const V3D& v3) : tag(MDVectorTag::V3), v3(v3) {}
    MDVector(const V4D& v4) : tag(MDVectorTag::V4), v4(v4) {}
    ~MDVector() {}

    MDVector(const MDVector& other) {
        v1 = other.v1;
        v2 = other.v2;
        v3 = other.v3;
        v4 = other.v4;
    }

    MDVector& operator=(const MDVector& other) {
        v1 = other.v1;
        v2 = other.v2;
        v3 = other.v3;
        v4 = other.v4;
        return *this;
    }

    MDVector& operator=(const V1D& other) {
        v1 = other;
        return *this;
    }

    MDVector& operator=(const V2D& other) {
        v2 = other;
        return *this;
    }

    MDVector& operator=(const V3D& other) {
        v3 = other;
        return *this;
    }

    MDVector& operator=(const V4D& other) {
        v4 = other;
        return *this;
    }
};