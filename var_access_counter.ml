open Cil_types
open Visitor

module VarAccessCounter = struct
  (* Hash table to store variable access counts *)
  let var_reads = Hashtbl.create 100
  let var_writes = Hashtbl.create 100
  
  (* Helper function to create unique variable identifier *)
  let make_var_id filename func_name var_name =
    Printf.sprintf "%s:%s::%s" filename func_name var_name
  
  (* Helper function to get filename from location *)
  let get_filename loc =
    let filename = (fst loc).Lexing.pos_fname in
    Filename.basename filename
  
  (* Helper function to increment counter *)
  let increment_counter table var_id =
    let count = try Hashtbl.find table var_id with Not_found -> 0 in
    Hashtbl.replace table var_id (count + 1)
  
  (* Record a variable read *)
  let record_read var_id =
    increment_counter var_reads var_id
  
  (* Record a variable write *)
  let record_write var_id =
    increment_counter var_writes var_id
  
  (* Visitor class for detecting variable accesses *)
  class var_access_visitor = object (self)
    inherit Visitor.frama_c_inplace
    
    (* Current function context *)
    val mutable current_function = "unknown"
    val mutable current_filename = "unknown"
    
    (* Visit function definitions *)
    method! vfunc fundec =
      current_function <- fundec.svar.vname;
      current_filename <- get_filename fundec.svar.vdecl;
      Kernel.feedback ~level:1 "[kernel] == %s:%s ==" current_filename current_function;
      Cil.DoChildren
    
    (* Visit expressions - handles variable reads *)
    method! vexpr exp =
      (match exp.enode with
       | Lval (Var vi, NoOffset) ->
         (* Variable read *)
         let var_id = make_var_id current_filename current_function vi.vname in
         record_read var_id
       | _ -> ());
      Cil.DoChildren
    
    (* Visit instructions - handles assignments and function calls *)
    method! vinst instr =
      (match instr with
       | Set (lv, exp, _) ->
         (* Assignment: lv = exp *)
         (match lv with
          | (Var vi, NoOffset) ->
            let var_id = make_var_id current_filename current_function vi.vname in
            record_write var_id
          | _ -> ());
         (* Process the expression for reads *)
         ignore (visitFramacExpr (self :> Visitor.frama_c_visitor) exp)
       | Call (Some lv, _, args, _) ->
         (* Function call with return value: lv = func(args) *)
         (match lv with
          | (Var vi, NoOffset) ->
            let var_id = make_var_id current_filename current_function vi.vname in
            record_write var_id
          | _ -> ());
         (* Process arguments for reads *)
         List.iter (fun arg -> 
           ignore (visitFramacExpr (self :> Visitor.frama_c_visitor) arg)) args
       | Call (None, _, args, _) ->
         (* Function call without return value: func(args) *)
         List.iter (fun arg -> 
           ignore (visitFramacExpr (self :> Visitor.frama_c_visitor) arg)) args
       | _ -> ());
      Cil.DoChildren
    
    (* Visit statements - handles return statements *)
    method! vstmt stmt =
      (match stmt.skind with
       | Return (Some exp, _) ->
         (* Return statement with expression *)
         ignore (visitFramacExpr (self :> Visitor.frama_c_visitor) exp)
       | _ -> ());
      Cil.DoChildren
    
    (* Visit variable declarations *)
    method! vvdec vi =
      (* Check if variable has initialization *)
      (match vi.vinit with
       | Some init ->
         let var_id = make_var_id current_filename current_function vi.vname in
         record_write var_id;
         (* Process initializer for reads *)
         (match init with
          | SingleInit exp ->
            ignore (visitFramacExpr (self :> Visitor.frama_c_visitor) exp)
          | _ -> ())
       | None -> ());
      Cil.DoChildren
  end
  
  (* Print results *)
  let print_results () =
    let all_vars = Hashtbl.fold (fun var_id _ acc -> var_id :: acc) var_reads [] in
    let all_vars = Hashtbl.fold (fun var_id _ acc -> 
      if List.mem var_id acc then acc else var_id :: acc) var_writes all_vars in
    
    List.iter (fun var_id ->
      let reads = try Hashtbl.find var_reads var_id with Not_found -> 0 in
      let writes = try Hashtbl.find var_writes var_id with Not_found -> 0 in
      Kernel.feedback ~level:1 "[kernel] %s: reads=%d  writes=%d" var_id reads writes
    ) (List.sort String.compare all_vars)
  
  (* Main analysis function *)
  let analyze_file () =
    let visitor = new var_access_visitor in
    Visitor.visitFramacFile (visitor :> Visitor.frama_c_visitor) (Ast.get ());
    print_results ()
end

(* Plugin registration *)
let () =
  Db.Main.extend (fun () ->
    VarAccessCounter.analyze_file ()
  )