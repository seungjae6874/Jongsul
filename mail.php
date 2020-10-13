<?php

    include "Sendmail.php";

    $to = 'litong4379@gmail.com';
    $firstname = $_POST["fname"];
    $email= $_POST["email"];
    $text= $_POST["message"];
    $phone= $_POST["phone"];



    $headers = 'MIME-Version: 1.0' . "\r\n";
    $headers .= "From: " . $email . "\r\n"; // Sender's E-mail
    $headers .= 'Content-type: text/html; charset=iso-8859-1' . "\r\n";

    $message ='<table style="width:100%">
        <tr>
            <td>'.$firstname.'  '.$laststname.'</td>
        </tr>
        <tr><td>Email: '.$email.'</td></tr>
        <tr><td>phone: '.$phone.'</td></tr>
        <tr><td>Text: '.$text.'</td></tr>

    </table>';

    if (@mail($to, $email, $message, $headers))
    {
        echo 'The message has been sent.';
    }else{
        echo 'failed';
    }

// 여기서부터 다시 시작
// include "Sendmail.php";

/*
+ $to : 받는사람 메일주소 ( ex. $to="hong <hgd@example.com>" 으로도 가능)
+ $from : 보내는사람 이름 + $subject : 메일 제목
+ $body : 메일 내용 + $cc_mail : Cc 메일 있을경우 (옵션값으로 생략가능)
+ $bcc_mail : Bcc 메일이 있을경우 (옵션값으로 생략가능) */
// $to="litong4379@gmail.com";
// $from="Master";
// $subject="메일 제목입니다.";
// $body="메일 내용입니다.";
// $cc_mail="cc@example.com";
// $bcc_mail="bcc@example.com";
//
//
//
//
// $sendmail = new Sendmail();


?>
