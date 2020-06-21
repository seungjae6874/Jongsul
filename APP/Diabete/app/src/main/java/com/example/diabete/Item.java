package com.example.diabete;

class Item {
    private String foodname;
    private int prot, fat,carbo;

    Item(String foodname, int prot, int fat, int carbo){
        foodname = foodname;
        prot = prot;
        fat = fat;
        carbo = carbo;
    }

    String getFoodname() {
        return foodname;
    }

    int getFat() {
        return fat;
    }

    int getProt() {
        return prot;
    }

    int getCarbo() {
        return carbo;
    }
}