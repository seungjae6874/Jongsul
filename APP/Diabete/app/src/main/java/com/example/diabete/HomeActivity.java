package com.example.diabete;

import android.Manifest;
import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;


public class HomeActivity extends AppCompatActivity {
    private ArrayList<Item> mItems = new ArrayList<>();
    private static final int WRITE_EXTERNAL_STORAGE_CODE = 1;
    EditText fat,carbo,protein, foodname;
    int ffat,fcarbo,fprotein,fname;

    Button search,save,create;
    String getcarbo,getprotein,getfat,getfoodname;
    int mLastRowNum;
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.home);

        search = findViewById(R.id.web);//webview로 음식 찾기
        save = findViewById(R.id.save);//엑셀에 저장하기 음식이름, 음식 고유번호
        fat = findViewById(R.id.fat);
        protein = findViewById(R.id.protein);
        carbo = findViewById(R.id.carbo);
        foodname = findViewById(R.id.foodname);
        create = findViewById(R.id.create);


        search.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(HomeActivity.this, MainActivity.class);
                startActivity(intent);
            }
        });
        //음식 번호, 이름 받아내기


        //엑셀 생성
        create.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getfoodname = foodname.getText().toString();
                getfat = fat.getText().toString();
                getcarbo = carbo.getText().toString();
                getprotein = protein.getText().toString();
                try{
                    //fname = Integer.parseInt(getfoodname);
                    ffat = Integer.parseInt(getfat);
                    fprotein = Integer.parseInt(getprotein);
                    fcarbo = Integer.parseInt(getcarbo);


                }catch (NumberFormatException e){

                }catch (Exception e){

                }

                createExcel();


                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                    if(checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)== PackageManager.PERMISSION_DENIED){
                        String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
                        //show popup for runtime permission
                        requestPermissions(permissions,WRITE_EXTERNAL_STORAGE_CODE);
                    }
                    else{//음식이름을 입력한 txt를 저장하고 나서 aws에 업로드하자.
                        //permission already granted
                        //saveExcel();
                        Toast.makeText(HomeActivity.this, String.valueOf(getExternalFilesDir(null)),Toast.LENGTH_LONG).show();
                        //Toast.makeText(HomeActivity.this, "엑셀 생성!",Toast.LENGTH_LONG).show();
                    }
                }
                else{
                    //saveExcel();
                    Toast.makeText(HomeActivity.this, "Success to upload",Toast.LENGTH_LONG).show();

                }

            }
        });


        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getfoodname = foodname.getText().toString();
                getfat = fat.getText().toString();
                getcarbo = carbo.getText().toString();
                getprotein = protein.getText().toString();
                try{
                    //fname = Integer.parseInt(getfoodname);
                    ffat = Integer.parseInt(getfat);
                    fprotein = Integer.parseInt(getprotein);
                    fcarbo = Integer.parseInt(getcarbo);


                }catch (NumberFormatException e){

                }catch (Exception e){

                }

                saveExcel();


                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                    if(checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)== PackageManager.PERMISSION_DENIED){
                        String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
                        //show popup for runtime permission
                        requestPermissions(permissions,WRITE_EXTERNAL_STORAGE_CODE);
                    }
                    else{//음식이름을 입력한 txt를 저장하고 나서 aws에 업로드하자.
                        //permission already granted
                        //saveExcel();

                        Toast.makeText(HomeActivity.this, String.valueOf(getExternalFilesDir(null)),Toast.LENGTH_LONG).show();
                    }
                }
                else{
                    //saveExcel();


                    Toast.makeText(HomeActivity.this, "Success to upload",Toast.LENGTH_LONG).show();

                }

            }


        });



    }
    private void initData(){
        mItems.add(new Item(getfoodname,fprotein,ffat,fcarbo));

    }

    //초기생성
    private void createExcel(){
        Workbook workbook = new HSSFWorkbook();

        Sheet sheet = workbook.createSheet("diabete"); // 새로운 시트 생성

        Row row = sheet.createRow(0); // 새로운 행 생성

        Cell cell;

        for(int i = 0; i < mItems.size() ; i++){ // 데이터 엑셀에 입력
            row = sheet.createRow(i+1);
            cell = row.createCell(0);
            cell.setCellValue(123);
            cell = row.createCell(1);
            cell.setCellValue(mItems.get(i).getProt());
            cell = row.createCell(2);
            cell.setCellValue(mItems.get(i).getFat());
            cell = row.createCell(3);
            cell.setCellValue(mItems.get(i).getCarbo());

        }

        // 데이터 엑셀에 입력
        row = sheet.createRow(0);
        cell = row.createCell(0);
        cell.setCellValue("Food");
        cell = row.createCell(1);
        cell.setCellValue("Protein");
        cell = row.createCell(2);
        cell.setCellValue("Fat");
        cell = row.createCell(3);
        cell.setCellValue("Carbo");
        cell = row.createCell(4);
        cell.setCellValue("Time");

        //값입력
        row = sheet.createRow(1);
        cell = row.createCell(0);
        cell.setCellValue(getfoodname);
        cell = row.createCell(1);
        cell.setCellValue(fprotein);
        cell = row.createCell(2);
        cell.setCellValue(ffat);
        cell = row.createCell(3);
        cell.setCellValue(fcarbo);
        cell = row.createCell(4);
        SimpleDateFormat format1 = new SimpleDateFormat("yyyy-MM-dd HH:mm");
        Date time = new Date();
        String time1 = format1.format(time);
        cell.setCellValue(time1); // 현재 날짜, 시 , 분 ,초



        File xlsFile = new File(getExternalFilesDir(null),"test.xls");
        try{
            FileOutputStream os = new FileOutputStream(xlsFile);
            workbook.write(os); // 외부 저장소에 엑셀 파일 생성
        }catch (IOException e){
            e.printStackTrace();
        }
        Toast.makeText(getApplicationContext(),xlsFile.getAbsolutePath()+"에 저장되었습니다",Toast.LENGTH_SHORT).show();
        Uri path = Uri.fromFile(xlsFile);
        Intent shareIntent = new Intent(Intent.ACTION_SEND);
        shareIntent.setType("application/excel");
        shareIntent.putExtra(Intent.EXTRA_STREAM,path);
        //startActivity(Intent.createChooser(shareIntent,"엑셀 내보내기"));


    }




    private void saveExcel() {
        try {
            String path = String.valueOf(getExternalFilesDir(null));
            File test = new File(getExternalFilesDir(null),"test.xls");
            FileInputStream fis = new FileInputStream(test);
            Workbook workbook = new HSSFWorkbook(fis);
            Sheet sheet = workbook.getSheetAt(0); // 해당 엑셀파일의 시트(Sheet) 수
            Log.d("sheet name :", sheet.getSheetName());
            int rows = sheet.getPhysicalNumberOfRows(); // 해당 시트의 행의 개수
            Log.d("행의 갯수  :", String.valueOf(rows));
            //값입력
            int prow = rows;
            Row row= sheet.createRow(prow);
            Cell cell;
            cell = row.createCell(0);
            cell.setCellValue(getfoodname);
            cell = row.createCell(1);
            cell.setCellValue(fprotein);
            cell = row.createCell(2);
            cell.setCellValue(ffat);
            cell = row.createCell(3);
            cell.setCellValue(fcarbo);
            cell = row.createCell(4);
            SimpleDateFormat format1 = new SimpleDateFormat("yyyy-MM-dd HH:mm");
            Date time = new Date();
            String time1 = format1.format(time);
            cell.setCellValue(time1); // 현재 날짜, 시 , 분 ,초

            //값을 넣었으면 이제 진짜로 excel에 집어넣자
            File xlsFile = new File(getExternalFilesDir(null),"test.xls");
            try{
                FileOutputStream os = new FileOutputStream(xlsFile);
                workbook.write(os); // 외부 저장소에 엑셀 파일 생성
            }catch (IOException e){
                e.printStackTrace();
            }
            Toast.makeText(getApplicationContext(),xlsFile.getAbsolutePath()+"갱신 되었습니다",Toast.LENGTH_SHORT).show();
            //Uri path = Uri.fromFile(xlsFile);
            //Intent shareIntent = new Intent(Intent.ACTION_SEND);
            //shareIntent.setType("application/excel");
            //shareIntent.putExtra(Intent.EXTRA_STREAM,path);
            //startActivity(Intent.createChooser(shareIntent,"엑셀 내보내기"));





        } catch (FileNotFoundException e){
            e.printStackTrace();
            Toast.makeText(HomeActivity.this, String.valueOf(getExternalFilesDir(null)),Toast.LENGTH_LONG ).show();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode){
            case WRITE_EXTERNAL_STORAGE_CODE:{
                //if request if cancelled,
                if(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED){
                    //saveExcel();
                }
                else{
                    Toast.makeText(HomeActivity.this,"Storage permissions ",Toast.LENGTH_SHORT).show();
                }
            }
        }
    }
}
