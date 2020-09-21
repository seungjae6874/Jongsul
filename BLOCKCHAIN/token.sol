pragma solidity ^0.4.18;

contract Token {
    function totalSupply() public view returns (uint256 supply){ }
    function balanceOf(address _owner) public view returns (uint256 balance){}
    function transfer(address _to, uint256 _value) public returns (bool success){}
    function transferFrom(address _from, address _to, uint256 _value) public returns(bool success){}
    function approve(address _spender, uint256 _value) public returns (bool success){}
    function allowance(address _owner, address _spender) public view returns (uint256 remaining){}
    
    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}

contract StandardToken is Token {
    function transfer(address _to, uint256 _value) public returns (bool success){
        if(balances[msg.sender] >= _value && _value >0){
            balances[msg.sender] -= _value;
            balances[_to] += _value;
            Transfer(msg.sender, _to, _value);
            return true;
        }else{ return false;}
    } 
    
    function balanceOf(address _owner) public view returns (uint256 balance){
        return balances[_owner];
    }
    
    mapping(address => uint256) balances;
    
}


contract DiabetesToken is StandardToken{
    string public name;
    uint8 public decimals; 
    string public symbol;
    string public version = 'H1.0';
    uint256 public Token_OneEthCanBuy;
    uint256 public totalEthInWei;
    address public fundManager;
    uint256 public totalSupply;
    
    function DiabetesToken(){
        balances[msg.sender] = 1000000000000000000; //1DBT
        totalSupply = 1000000000000000000; //1DBT
        name = "DiabetesToken";
        decimals = 18;
        symbol = "DBT";
        Token_OneEthCanBuy = 10;
        fundManager = msg.sender;
    }
    
    function() external payable { //여기_수정해야한다_정현아
        totalEthInWei = totalEthInWei + msg.value;
        uint256 amount = msg.value * Token_OneEthCanBuy;
        require(balances[fundManager] >= amount);

        
        balances[fundManager] = balances[fundManager] - amount;
        balances[msg.sender] = balances[msg.sender] + amount;
        
        Transfer(fundManager, msg.sender, amount);
        
        fundManager.transfer(msg.value);
    }
    

}
