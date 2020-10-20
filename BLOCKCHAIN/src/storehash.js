/* eslint no-eval: 0 */
import web3 from './web3';
import StoreHash from './StoreHash.json';


const address = JSON.stringify(StoreHash.networks["5777"].address);

//console.log(address);

var abi = eval(JSON.stringify(StoreHash.abi));

//console.log(abi);


export default new web3.eth.Contract(abi, address.replace(/"/g,""));
